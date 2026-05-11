from src.llms.ollama_llm import OllamaLLM


def main():
    llm = OllamaLLM(
        model_name="qwen2.5:3b",
        temperature=0.7,
        max_tokens=128,
    )

    response = llm.generate(
        prompt="Give one short argument in favor of banning political misinformation on social media.",
        stance="pro",
        topic="Should social media platforms ban political misinformation?",
        round_number=1,
    )

    print("\nOllama response:")
    print(response)


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
