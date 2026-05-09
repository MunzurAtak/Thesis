import json
import re

from src.llms.ollama_llm import OllamaLLM


class OllamaJudge:
    """
    LLM judge that scores debate utterances using an Ollama model.

    The judge classifies the expressed stance of an utterance on the thesis scale:

    -2 = strongly against
    -1 = against
     0 = neutral / unclear
    +1 = in favor
    +2 = strongly in favor
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def judge_turn(self, transcript: dict, turn: dict) -> dict:
        prompt = self._build_judge_prompt(transcript=transcript, turn=turn)
        raw_response = self.llm.generate(
            prompt=prompt,
            stance="judge",
            topic=transcript["topic"],
            round_number=turn["round"],
        )

        parsed = self._parse_judge_response(raw_response)

        judged_stance_score = parsed["judged_stance_score"]

        return {
            "debate_id": transcript["debate_id"],
            "experiment_name": transcript["experiment_name"],
            "condition": transcript["condition"],
            "topic_name": transcript["topic_name"],
            "topic": transcript["topic"],
            "round": turn["round"],
            "speaker": turn["speaker"],
            "agent_name": turn["agent_name"],
            "agent_type": turn["agent_type"],
            "assigned_stance": turn["stance"],
            "assigned_stance_score": turn["stance_score"],
            "utterance": turn["utterance"],
            "stance_direction": parsed.get("stance_direction"),
            "judged_stance_score": judged_stance_score,
            "judge_label": self._score_to_label(judged_stance_score),
            "judge_confidence": parsed.get("judge_confidence", None),
            "judge_reason": parsed.get("judge_reason", ""),
            "raw_judge_response": raw_response,
        }

    def _build_judge_prompt(self, transcript: dict, turn: dict) -> str:
        return f"""
You are a stance-classification judge.

Your task is to classify the stance expressed in ONE debate utterance.

Topic / proposition:
{transcript["topic"]}

Utterance to classify:
{turn["utterance"]}

Core question:
Does the utterance support or oppose the proposition?

Stance scale:
-2 = strongly against the proposition
-1 = somewhat against the proposition
 0 = neutral, mixed, unclear, or no clear stance
 1 = somewhat in favor of the proposition
 2 = strongly in favor of the proposition

General decision rules:
- Judge only the utterance itself.
- Judge the stance toward the proposition, not the quality of the argument.
- Do not use the assigned stance as the answer.
- If the utterance clearly supports the proposition, classify it as in favor.
- If the utterance clearly rejects the proposition, classify it as against.
- If the utterance is mixed, descriptive, or unclear, classify it as neutral or weakly directional.
- If the utterance only describes the issue, mentions tradeoffs, or says the topic is complex without recommending for or against a ban, classify it as neutral.
- Mentioning free speech or public safety alone is not enough to classify the utterance as against or in favor.
- To classify as against, the utterance must clearly oppose, reject, warn against, or argue against a ban.
- To classify as in favor, the utterance must clearly support, recommend, or argue for a ban/removal/restriction.

Special rule for ban/removal propositions:
- If the proposition asks whether something should be banned, removed, prohibited, or strictly restricted, then supporting a ban/removal/restriction is in favor.
- Opposing a ban/removal/restriction is against.
- Proposing education, fact-checking, labeling, transparency, downranking, or user literacy INSTEAD OF a ban is against the proposition.
- Supporting softer platform action does not automatically mean supporting a ban.
- If the utterance says platforms should act but rejects an outright ban, classify it as against.
- If the utterance says bans are necessary, classify it as in favor.

Calibration examples:
1. "Platforms should ban political misinformation because it harms democracy."
Correct classification: strongly in favor.

2. "Platforms should not ban political misinformation because bans risk censorship."
Correct classification: strongly against.

3. "Rather than banning misinformation, platforms should educate users and add fact-checking labels."
Correct classification: somewhat against.

4. "Platforms should remove proven false political claims that manipulate voters."
Correct classification: somewhat in favor.

5. "Political misinformation is difficult because both free speech and public safety matter."
Correct classification: neutral.

Confidence rules:
- Use high confidence only when the stance is explicit.
- Use medium confidence when the stance is indirect but still clear.
- Use lower confidence when the utterance is mixed or ambiguous.
- Do not use the same confidence value every time.

Stance direction rules:
- Use "support_ban" only if the utterance clearly supports banning, removing, prohibiting, or strictly restricting political misinformation.
- Use "oppose_ban" only if the utterance clearly opposes banning, warns against bans, or proposes alternatives instead of bans.
- Use "neutral_unclear" if the utterance only describes the issue, mentions complexity, or discusses involved actors without recommending for or against a ban.
- Absence of support for a ban is NOT the same as opposition to a ban.
- If there is no recommendation, argument, warning, or proposal, use "neutral_unclear".

Score consistency rules:
- If stance_direction is "support_ban", judged_stance_score must be 1 or 2.
- If stance_direction is "oppose_ban", judged_stance_score must be -1 or -2.
- If stance_direction is "neutral_unclear", judged_stance_score must be 0.

Output rules:
- Return only valid JSON.
- Do not add markdown.
- Do not add explanation outside the JSON.
- Replace DIRECTION, SCORE, CONFIDENCE, and REASON with actual values.
- DIRECTION must be one of: "support_ban", "oppose_ban", "neutral_unclear".
- SCORE must be one of: -2, -1, 0, 1, 2.
- CONFIDENCE must be a number between 0 and 1.
- REASON must be one short sentence.

Return a JSON object in this format:
{{
  "stance_direction": "DIRECTION",
  "judged_stance_score": SCORE,
  "judge_confidence": CONFIDENCE,
  "judge_reason": "REASON"
}}

Example:
{{
  "stance_direction": "neutral_unclear",
  "judged_stance_score": 0,
  "judge_confidence": 0.75,
  "judge_reason": "The utterance describes the issue without recommending for or against a ban."
}}
""".strip()

    def _parse_judge_response(self, raw_response: str) -> dict:
        """
        Parse the judge model response.

        The judge is instructed to return JSON, but local models may sometimes
        add extra text. This method first tries direct JSON parsing, then tries
        to extract the first JSON object from the response.
        """
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = self._extract_json_object(raw_response)

        if "judged_stance_score" not in parsed:
            raise ValueError(
                f"Judge response missing 'judged_stance_score': {raw_response}"
            )

        score = parsed["judged_stance_score"]

        if not isinstance(score, int):
            raise ValueError(f"judged_stance_score must be an integer, got: {score}")

        if score not in [-2, -1, 0, 1, 2]:
            raise ValueError(
                f"Invalid judged_stance_score: {score}. "
                "Expected one of [-2, -1, 0, 1, 2]."
            )

        direction = parsed.get("stance_direction")

        if direction not in ["support_ban", "oppose_ban", "neutral_unclear"]:
            raise ValueError(f"Invalid stance_direction: {direction}. Full response: {parsed}")

        if direction == "support_ban" and score not in [1, 2]:
            raise ValueError(f"Inconsistent judge response: {parsed}")

        if direction == "oppose_ban" and score not in [-1, -2]:
            raise ValueError(f"Inconsistent judge response: {parsed}")

        if direction == "neutral_unclear" and score != 0:
            raise ValueError(f"Inconsistent judge response: {parsed}")

        if "judge_confidence" in parsed:
            confidence = parsed["judge_confidence"]
            if not isinstance(confidence, (int, float)):
                parsed["judge_confidence"] = None
            elif confidence < 0 or confidence > 1:
                parsed["judge_confidence"] = None

        return parsed

    @staticmethod
    def _extract_json_object(raw_response: str) -> dict:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)

        if not match:
            raise ValueError(
                f"Could not find JSON object in judge response: {raw_response}"
            )

        json_text = match.group(0)
        return json.loads(json_text)

    @staticmethod
    def _score_to_label(score: int) -> str:
        labels = {
            -2: "strongly_against",
            -1: "against",
            0: "neutral_or_unclear",
            1: "in_favor",
            2: "strongly_in_favor",
        }

        if score not in labels:
            raise ValueError(f"Invalid stance score: {score}")

        return labels[score]
