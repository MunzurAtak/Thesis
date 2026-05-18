import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a QLoRA stance adapter for Qwen2.5-7B-Instruct."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base Hugging Face model name.",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default="data/lora_training/usdc_3topics_controlled_lora_train.jsonl",
        help="Path to controlled LoRA train JSONL.",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        default="data/lora_training/usdc_3topics_controlled_lora_val.jsonl",
        help="Path to controlled LoRA validation JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/lora_adapters/qwen2_5_7b_stance_lora",
        help="Directory where the LoRA adapter will be saved.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "messages" not in row:
                raise ValueError(
                    f"Missing messages field at line {line_number}: {path}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No rows loaded from: {path}")

    return rows


def format_and_tokenize(example: dict, tokenizer, max_length: int) -> dict:
    messages = example["messages"]

    if len(messages) < 3:
        raise ValueError("Expected at least system, user, and assistant messages.")

    prompt_messages = messages[:-1]
    assistant_message = messages[-1]

    if assistant_message.get("role") != "assistant":
        raise ValueError("Final message must be the assistant response.")

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assistant_text = assistant_message["content"]

    if tokenizer.eos_token:
        assistant_text = assistant_text + tokenizer.eos_token

    full_text = prompt_text + assistant_text

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = tokenized_full["input_ids"].copy()
    prompt_length = len(tokenized_prompt["input_ids"])

    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


def main():
    args = parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    print(f"Loading train data from: {train_path}")
    train_rows = load_jsonl(train_path)

    print(f"Loading validation data from: {val_path}")
    val_rows = load_jsonl(val_path)

    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing datasets...")
    train_dataset = Dataset.from_list(train_rows).map(
        lambda row: format_and_tokenize(row, tokenizer, args.max_length),
        remove_columns=list(train_rows[0].keys()),
    )

    val_dataset = Dataset.from_list(val_rows).map(
        lambda row: format_and_tokenize(row, tokenizer, args.max_length),
        remove_columns=list(val_rows[0].keys()),
    )

    print("Loading base model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print("Adding LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete.")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
