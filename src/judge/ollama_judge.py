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

Stance scale:
-2 = strongly against the proposition
-1 = somewhat against the proposition
 0 = neutral, mixed, unclear, or no clear stance
 1 = somewhat in favor of the proposition
 2 = strongly in favor of the proposition

Decision rules:
- The proposition is specifically whether social media platforms SHOULD BAN political misinformation.
- If the utterance supports banning, removing, prohibiting, or strongly restricting political misinformation, classify it as in favor.
- If the utterance opposes bans, argues bans are censorship, defends free speech, or proposes education/fact-checking/transparency as an alternative to banning, classify it as against.
- Mentioning fact-checking, transparency, education, or moderation does NOT automatically mean the utterance supports a ban.
- If the utterance says platforms should act but rejects an outright ban, classify it as against.
- If the utterance clearly says bans are necessary, classify it as in favor.
- Judge only the expressed stance toward the proposition, not the quality of the argument.
- Do not use the assigned stance as the answer.

Output rules:
- Return only valid JSON.
- Do not add markdown.
- Do not add explanation outside the JSON.
- Do not copy example values.
- Choose the score and confidence based only on the utterance.

Required JSON fields:
- judged_stance_score: integer, one of -2, -1, 0, 1, 2
- judge_confidence: number between 0 and 1
- judge_reason: one short sentence

Replace SCORE, CONFIDENCE, and REASON with actual values.

Return a JSON object in this format:
{{
  "judged_stance_score": SCORE,
  "judge_confidence": CONFIDENCE,
  "judge_reason": "REASON"
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
