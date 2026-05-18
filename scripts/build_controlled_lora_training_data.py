import argparse
import json
import random
from pathlib import Path

TOPICS = {
    "climate_change": {
        "question": "Should governments take stronger action against climate change?",
        "stances": {
            "pro": [
                {
                    "angle": "emissions regulation",
                    "claim": "Governments should take stronger action by setting stricter emissions standards.",
                    "reason": "greenhouse gas emissions create long-term environmental and economic risks that individual choices alone cannot solve",
                    "policy": "clear emission limits can push industries toward cleaner production while giving them predictable rules",
                },
                {
                    "angle": "renewable energy",
                    "claim": "Governments should invest more aggressively in renewable energy.",
                    "reason": "public investment can accelerate the transition away from fossil fuels and reduce future climate damage",
                    "policy": "subsidies, infrastructure spending, and grid modernization can make cleaner energy more practical",
                },
                {
                    "angle": "carbon pricing",
                    "claim": "Governments should use carbon pricing to make polluters account for environmental costs.",
                    "reason": "markets often ignore the long-term damage caused by emissions",
                    "policy": "a carbon fee can encourage cleaner alternatives while funding support for affected households",
                },
                {
                    "angle": "public health",
                    "claim": "Stronger government climate action is justified because climate change harms public health.",
                    "reason": "heat waves, air pollution, food disruption, and extreme weather place direct burdens on citizens",
                    "policy": "preventive regulation can reduce future health costs and protect vulnerable communities",
                },
                {
                    "angle": "intergenerational responsibility",
                    "claim": "Governments have a responsibility to act now because future generations will bear the costs of delay.",
                    "reason": "waiting makes climate impacts harder and more expensive to manage",
                    "policy": "early mitigation is more responsible than leaving irreversible damage to future citizens",
                },
                {
                    "angle": "coordination problem",
                    "claim": "Climate change requires stronger government action because it is a collective-action problem.",
                    "reason": "private actors have little incentive to reduce emissions if competitors can continue polluting",
                    "policy": "government rules can coordinate action across industries and prevent free-riding",
                },
                {
                    "angle": "economic transition",
                    "claim": "Stronger climate policy can support long-term economic stability.",
                    "reason": "uncontrolled climate damage threatens agriculture, infrastructure, insurance markets, and public budgets",
                    "policy": "planned investment in cleaner industries can reduce disruption and create new economic opportunities",
                },
                {
                    "angle": "international leadership",
                    "claim": "Governments should act more strongly to support international climate cooperation.",
                    "reason": "credible domestic policy makes global agreements more realistic",
                    "policy": "countries that reduce emissions can pressure others to follow and strengthen international standards",
                },
            ],
            "contra": [
                {
                    "angle": "economic burden",
                    "claim": "Governments should not impose stronger climate policies too aggressively.",
                    "reason": "rapid regulation can raise energy prices, harm workers, and place pressure on households",
                    "policy": "climate policy should be gradual and economically realistic rather than disruptive",
                },
                {
                    "angle": "private innovation",
                    "claim": "Private-sector innovation is a better path than stronger government intervention.",
                    "reason": "companies and researchers can develop cleaner technology without broad regulatory overreach",
                    "policy": "governments should support innovation without imposing heavy restrictions on the economy",
                },
                {
                    "angle": "adaptation first",
                    "claim": "Governments should prioritize adaptation rather than stronger mitigation mandates.",
                    "reason": "some climate impacts are already unavoidable and communities need practical protection",
                    "policy": "flood defenses, resilient infrastructure, and emergency planning may be more effective than strict emission rules",
                },
                {
                    "angle": "policy uncertainty",
                    "claim": "Stronger government climate action can be ineffective if policies are poorly designed.",
                    "reason": "large bureaucratic programs may waste resources or create unintended consequences",
                    "policy": "targeted and evidence-based measures are preferable to sweeping intervention",
                },
                {
                    "angle": "global coordination limits",
                    "claim": "Unilateral government action may have limited effect without global cooperation.",
                    "reason": "emissions are a global problem and strict rules in one country may simply shift production elsewhere",
                    "policy": "governments should avoid policies that damage domestic industries without meaningfully lowering global emissions",
                },
                {
                    "angle": "energy security",
                    "claim": "Governments should be cautious about stronger climate action that weakens energy security.",
                    "reason": "a rapid transition away from reliable energy sources can create shortages and instability",
                    "policy": "climate goals should be balanced with affordability, reliability, and national resilience",
                },
                {
                    "angle": "market flexibility",
                    "claim": "Market-based and voluntary approaches can be better than stronger regulation.",
                    "reason": "rigid government rules may prevent flexible solutions that fit local conditions",
                    "policy": "incentives and innovation prizes can encourage cleaner behavior without excessive control",
                },
                {
                    "angle": "social tradeoffs",
                    "claim": "Stronger climate action should not be treated as automatically justified.",
                    "reason": "climate policy competes with other urgent priorities such as housing, healthcare, and poverty reduction",
                    "policy": "governments should balance environmental goals against broader social needs",
                },
            ],
        },
    },
    "abortion": {
        "question": "Should abortion remain legally accessible?",
        "stances": {
            "pro": [
                {
                    "angle": "bodily autonomy",
                    "claim": "Abortion should remain legally accessible because bodily autonomy is central to individual freedom.",
                    "reason": "pregnancy directly affects a person's body, health, and future",
                    "policy": "the law should allow people to make reproductive decisions with medical guidance",
                },
                {
                    "angle": "health and safety",
                    "claim": "Legal abortion access protects health and safety.",
                    "reason": "restricting access can push people toward unsafe procedures or delay urgent care",
                    "policy": "regulated medical access is safer than forcing abortion into hidden or unregulated settings",
                },
                {
                    "angle": "medical necessity",
                    "claim": "Abortion should remain legally accessible because some pregnancies involve serious medical risks.",
                    "reason": "complications can threaten the life or long-term health of the pregnant person",
                    "policy": "doctors need legal space to provide necessary care without fear of punishment",
                },
                {
                    "angle": "social inequality",
                    "claim": "Restricting abortion access often harms disadvantaged people most.",
                    "reason": "wealthier individuals can travel for care while poorer individuals face greater barriers",
                    "policy": "legal access helps prevent reproductive healthcare from depending on income or location",
                },
                {
                    "angle": "personal circumstances",
                    "claim": "Abortion should remain legally accessible because pregnancy decisions involve complex personal circumstances.",
                    "reason": "poverty, abuse, health, age, and family stability can all shape the decision",
                    "policy": "the law should not force one outcome for every case",
                },
                {
                    "angle": "public health",
                    "claim": "Maintaining legal abortion access is better for public health.",
                    "reason": "medical systems can provide counseling, screening, and safe procedures when care is legal",
                    "policy": "legal access allows standards, oversight, and follow-up care",
                },
                {
                    "angle": "privacy",
                    "claim": "Abortion should remain legally accessible because reproductive decisions are deeply private.",
                    "reason": "state control over pregnancy can intrude into intimate medical and family decisions",
                    "policy": "legal access preserves a private zone for patients and clinicians",
                },
                {
                    "angle": "practical consequences",
                    "claim": "Banning or severely restricting abortion does not remove the demand for abortion.",
                    "reason": "it mainly changes whether the procedure happens safely or dangerously",
                    "policy": "legal access is a practical way to reduce harm while respecting individual choice",
                },
            ],
            "contra": [
                {
                    "angle": "fetal life",
                    "claim": "Abortion should not remain broadly legally accessible because fetal life has moral value.",
                    "reason": "ending a pregnancy can mean ending a developing human life",
                    "policy": "the law should protect unborn life rather than treating abortion as an ordinary choice",
                },
                {
                    "angle": "sanctity of life",
                    "claim": "Legal abortion access should be restricted because society has a duty to protect vulnerable life.",
                    "reason": "the unborn cannot defend their own interests",
                    "policy": "laws should reflect a strong presumption in favor of preserving life",
                },
                {
                    "angle": "alternatives",
                    "claim": "Abortion should be restricted when alternatives such as adoption and social support are possible.",
                    "reason": "difficult circumstances do not automatically justify ending fetal life",
                    "policy": "governments should expand support for pregnant people and families instead of preserving broad abortion access",
                },
                {
                    "angle": "moral responsibility",
                    "claim": "Abortion access should not be treated as a default solution to unwanted pregnancy.",
                    "reason": "sexual decisions can create responsibilities toward a developing child",
                    "policy": "law should encourage responsibility while providing support for hardship cases",
                },
                {
                    "angle": "late pregnancy concerns",
                    "claim": "Legal abortion access raises especially serious concerns later in pregnancy.",
                    "reason": "the moral weight of fetal development increases as pregnancy progresses",
                    "policy": "the law should impose stronger restrictions as fetal development advances",
                },
                {
                    "angle": "social values",
                    "claim": "Restricting abortion can express society's commitment to protecting life.",
                    "reason": "law shapes moral norms as well as individual options",
                    "policy": "legal rules should discourage abortion while supporting women and children",
                },
                {
                    "angle": "rights conflict",
                    "claim": "Abortion should be restricted because the issue involves more than one set of interests.",
                    "reason": "bodily autonomy matters, but the potential life of the unborn also deserves legal consideration",
                    "policy": "the law should balance these interests rather than prioritizing access alone",
                },
                {
                    "angle": "ethical caution",
                    "claim": "When there is moral uncertainty about fetal status, the law should be cautious about abortion access.",
                    "reason": "if the unborn has significant moral status, abortion may cause irreversible harm",
                    "policy": "restrictions can reflect ethical caution in the face of uncertainty",
                },
            ],
        },
    },
    "gun_control": {
        "question": "Should governments implement stricter gun control laws?",
        "stances": {
            "pro": [
                {
                    "angle": "background checks",
                    "claim": "Governments should implement stricter gun control laws through stronger background checks.",
                    "reason": "firearms can cause severe harm when obtained by people with violent histories",
                    "policy": "universal checks can reduce risky access while still allowing lawful ownership",
                },
                {
                    "angle": "licensing",
                    "claim": "Firearm licensing is a reasonable form of stricter gun control.",
                    "reason": "guns require responsibility and training because misuse can be deadly",
                    "policy": "licensing can set minimum standards for safe ownership",
                },
                {
                    "angle": "red flag laws",
                    "claim": "Governments should use red flag laws to prevent foreseeable gun violence.",
                    "reason": "some risks are visible before a shooting occurs",
                    "policy": "temporary removal procedures with due process can protect public safety",
                },
                {
                    "angle": "public safety",
                    "claim": "Stricter gun control laws are justified by the need to reduce preventable deaths.",
                    "reason": "easy firearm access can make conflicts, accidents, and crises more lethal",
                    "policy": "reasonable restrictions can lower risks without banning all guns",
                },
                {
                    "angle": "mass shootings",
                    "claim": "Governments should restrict access to weapons and accessories that increase mass-casualty risks.",
                    "reason": "high-capacity magazines and rapid-fire weapons can make attacks more deadly",
                    "policy": "targeted restrictions can reduce the scale of harm when violence occurs",
                },
                {
                    "angle": "responsible ownership",
                    "claim": "Stricter gun laws can support responsible ownership rather than eliminate it.",
                    "reason": "responsible owners benefit when dangerous access and unsafe storage are reduced",
                    "policy": "training, storage rules, and licensing can create a safer ownership culture",
                },
                {
                    "angle": "domestic violence",
                    "claim": "Gun control is important in situations involving domestic violence or credible threats.",
                    "reason": "the presence of a firearm can escalate abuse into fatal violence",
                    "policy": "laws can restrict access for individuals with serious risk indicators",
                },
                {
                    "angle": "balanced regulation",
                    "claim": "Governments can implement stricter gun control while respecting lawful ownership.",
                    "reason": "rights can coexist with safety rules when dangerous misuse is foreseeable",
                    "policy": "background checks, licensing, and red flag laws are targeted rather than absolute bans",
                },
            ],
            "contra": [
                {
                    "angle": "self defense",
                    "claim": "Governments should not implement stricter gun control laws because firearms can be necessary for self-defense.",
                    "reason": "people may need effective protection before police can arrive",
                    "policy": "laws should not make lawful self-defense harder for responsible citizens",
                },
                {
                    "angle": "gun rights",
                    "claim": "Stricter gun control risks infringing on the rights of law-abiding citizens.",
                    "reason": "broad restrictions can punish responsible owners for the actions of criminals",
                    "policy": "the law should focus on offenders rather than burdening lawful ownership",
                },
                {
                    "angle": "enforce existing laws",
                    "claim": "Governments should enforce existing gun laws instead of adding stricter ones.",
                    "reason": "new restrictions may not help if current laws are not applied effectively",
                    "policy": "better enforcement and prosecution can target dangerous individuals more directly",
                },
                {
                    "angle": "black market",
                    "claim": "Stricter gun control may push firearms further into illegal markets.",
                    "reason": "criminals can ignore regulations while lawful owners comply",
                    "policy": "policy should address illegal trafficking rather than only restricting legal buyers",
                },
                {
                    "angle": "civil liberties",
                    "claim": "Stricter gun control can create civil-liberty problems.",
                    "reason": "licensing, registries, and confiscation procedures can be misused",
                    "policy": "public safety should be pursued without giving the state excessive power over lawful citizens",
                },
                {
                    "angle": "root causes",
                    "claim": "Gun violence should be addressed through root causes rather than stricter gun laws alone.",
                    "reason": "poverty, mental health, gangs, and social breakdown contribute to violence",
                    "policy": "community intervention and targeted policing may work better than broad restrictions",
                },
                {
                    "angle": "rural needs",
                    "claim": "Uniform stricter gun laws may ignore differences between communities.",
                    "reason": "rural citizens may rely on firearms for protection, hunting, or property defense",
                    "policy": "local conditions should matter more than one-size-fits-all national restrictions",
                },
                {
                    "angle": "ineffectiveness",
                    "claim": "Stricter gun control should not be assumed effective without clear evidence.",
                    "reason": "some places with strict rules still experience serious gun crime",
                    "policy": "governments should avoid symbolic laws and focus on measures with proven effects",
                },
            ],
        },
    },
}


TEMPLATES = [
    "{claim} This matters because {reason}. Therefore, {policy}.",
    "{claim} The central point is that {reason}. A better policy approach is that {policy}.",
    "{claim} If we take the issue seriously, we must recognize that {reason}. For that reason, {policy}.",
    "{claim} The strongest reason is that {reason}. In practice, {policy}.",
    "{claim} This stance does not require ignoring tradeoffs; it means recognizing that {reason}. As a result, {policy}.",
    "{claim} The opposing side often underestimates that {reason}. The more responsible conclusion is that {policy}.",
    "{claim} From a policy perspective, {reason}. This supports the view that {policy}.",
    "{claim} A consistent position on this topic starts from the fact that {reason}. Therefore, {policy}.",
]


STYLE_PREFIXES = [
    "",
    "",
    "My argument is straightforward: ",
    "The core point is this: ",
]


STANCE_TO_SCORE = {
    "pro": 2,
    "contra": -2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a controlled LoRA stance-conditioning dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lora_training",
    )

    parser.add_argument(
        "--train-output-name",
        type=str,
        default="usdc_3topics_controlled_lora_train.jsonl",
    )

    parser.add_argument(
        "--val-output-name",
        type=str,
        default="usdc_3topics_controlled_lora_val.jsonl",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def build_system_message(stance: str) -> str:
    return (
        "You are a debate agent in a controlled thesis experiment. "
        f"You must consistently argue from the assigned {stance} stance. "
        "Do not switch sides. Do not say that you are neutral. "
        "Do not include insults, profanity, personal attacks, or off-topic claims. "
        "Give a clear, concise argument that supports your assigned stance."
    )


def build_user_message(topic: str, stance: str) -> str:
    stance_description = (
        "in favor of the proposition" if stance == "pro" else "against the proposition"
    )

    return (
        f"Topic / proposition: {topic}\n\n"
        f"Assigned stance: {stance} ({stance_description}).\n\n"
        "Write one debate argument that is consistent with the assigned stance."
    )


def build_response(angle_data: dict, template: str, prefix: str) -> str:
    response = template.format(
        claim=angle_data["claim"],
        reason=angle_data["reason"],
        policy=angle_data["policy"],
    )

    return f"{prefix}{response}".strip()


def build_examples() -> list[dict]:
    examples = []

    for topic_name, topic_data in TOPICS.items():
        topic = topic_data["question"]

        for stance, angles in topic_data["stances"].items():
            for angle_data in angles:
                for template_index, template in enumerate(TEMPLATES):
                    for prefix_index, prefix in enumerate(STYLE_PREFIXES):
                        response = build_response(angle_data, template, prefix)

                        examples.append(
                            {
                                "topic_name": topic_name,
                                "topic": topic,
                                "stance": stance,
                                "stance_score": STANCE_TO_SCORE[stance],
                                "source": "controlled_synthetic",
                                "angle": angle_data["angle"],
                                "template_index": template_index,
                                "prefix_index": prefix_index,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": build_system_message(stance),
                                    },
                                    {
                                        "role": "user",
                                        "content": build_user_message(topic, stance),
                                    },
                                    {
                                        "role": "assistant",
                                        "content": response,
                                    },
                                ],
                            }
                        )

    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    examples = build_examples()
    random.shuffle(examples)

    val_size = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_output_name
    val_path = output_dir / args.val_output_name

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    print(f"Saved train data to: {train_path}")
    print(f"Saved validation data to: {val_path}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Total examples: {len(examples)}")

    print("\nExamples by topic and stance:")
    counts = {}

    for example in examples:
        key = (example["topic_name"], example["stance"])
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        topic_name, stance = key
        print(f"- {topic_name} / {stance}: {counts[key]}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
