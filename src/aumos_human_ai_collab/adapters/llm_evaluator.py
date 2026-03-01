"""LLM-as-judge evaluation adapter for aumos-human-ai-collab.

Uses a secondary LLM to evaluate the quality of primary model outputs on
configurable criteria (correctness, relevance, safety, coherence) before
they reach human reviewers. Results are stored as HacLLMEvaluationResult
records to help reviewers prioritise their attention.

Gap Coverage: GAP-257 (LLM-as-Judge Evaluation)
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Default evaluation criteria with their relative weights
_DEFAULT_CRITERIA_WEIGHTS: dict[str, float] = {
    "correctness": 0.35,
    "relevance": 0.30,
    "safety": 0.25,
    "coherence": 0.10,
}

# Evaluation prompt template — instructs the judge LLM
_EVALUATION_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of an AI-generated response.

Evaluate the following AI output on these criteria, scoring each from 0.0 to 1.0:
{criteria_list}

Task context:
{task_context}

AI Output to evaluate:
{ai_output}

Return a JSON object with this exact structure:
{{
  "criterion_scores": {{
    "<criterion_name>": <score_0.0_to_1.0>,
    ...
  }},
  "reasoning": "<brief explanation of scores>"
}}

Return only the JSON object, no other text."""


@dataclass
class EvaluationRequest:
    """Parameters for an LLM evaluation request.

    Attributes:
        hitl_review_id: UUID of the HITLReview to evaluate.
        ai_output: The AI-generated content to assess.
        task_context: Context about the task (task_type, prompt summary, etc.).
        criteria: List of criterion names to evaluate. Uses defaults if empty.
        criteria_weights: Optional per-criterion weight overrides.
    """

    hitl_review_id: uuid.UUID
    ai_output: str
    task_context: str
    criteria: list[str]
    criteria_weights: dict[str, float]


@dataclass
class EvaluationResult:
    """Result of an LLM-as-judge evaluation.

    Attributes:
        hitl_review_id: UUID of the evaluated HITLReview.
        judge_model_id: Model ID of the judge LLM used.
        criterion_scores: Dict of criterion_name -> score (0.0-1.0).
        composite_score: Weighted mean of criterion scores.
        judge_reasoning: Free-text reasoning from the judge.
        evaluation_latency_ms: Time taken to run the evaluation.
        flagged_for_review: True if composite_score < pass_threshold.
        evaluation_metadata: Additional context (token counts, etc.).
    """

    hitl_review_id: uuid.UUID
    judge_model_id: str
    criterion_scores: dict[str, float]
    composite_score: float
    judge_reasoning: str | None
    evaluation_latency_ms: int
    flagged_for_review: bool
    evaluation_metadata: dict[str, Any]


class LLMEvaluator:
    """LLM-as-judge evaluator for AI output quality assessment.

    Calls a secondary LLM (the judge) to score AI outputs on configurable
    criteria. Uses structured JSON output parsing for reliable score extraction.

    Args:
        judge_model_id: Model ID for the judge LLM (e.g., claude-sonnet-4-6).
        llm_api_base_url: Base URL of the LLM serving endpoint.
        llm_api_key: API key for the LLM serving endpoint.
        pass_threshold: Composite score below which output is flagged (default: 0.70).
        http_timeout_seconds: HTTP timeout for judge LLM calls.
        default_criteria_weights: Per-criterion weight overrides.
    """

    def __init__(
        self,
        judge_model_id: str,
        llm_api_base_url: str,
        llm_api_key: str,
        pass_threshold: float = 0.70,
        http_timeout_seconds: float = 30.0,
        default_criteria_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the LLM evaluator.

        Args:
            judge_model_id: Model ID for the judge LLM.
            llm_api_base_url: LLM API base URL.
            llm_api_key: LLM API authentication key.
            pass_threshold: Score below which output is flagged for priority review.
            http_timeout_seconds: HTTP timeout for LLM calls.
            default_criteria_weights: Custom weights (uses _DEFAULT_CRITERIA_WEIGHTS if None).
        """
        self._judge_model_id = judge_model_id
        self._llm_api_base_url = llm_api_base_url.rstrip("/")
        self._llm_api_key = llm_api_key
        self._pass_threshold = pass_threshold
        self._http_timeout = http_timeout_seconds
        self._criteria_weights = default_criteria_weights or dict(_DEFAULT_CRITERIA_WEIGHTS)
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Create the shared HTTP client.

        Must be called before evaluate() is invoked.
        """
        self._client = httpx.AsyncClient(
            base_url=self._llm_api_base_url,
            headers={
                "Authorization": f"Bearer {self._llm_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._http_timeout),
        )
        logger.info(
            "LLMEvaluator initialized",
            judge_model_id=self._judge_model_id,
            pass_threshold=self._pass_threshold,
        )

    async def close(self) -> None:
        """Close the HTTP client on shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate an AI output using the configured judge LLM.

        Sends a structured prompt to the judge LLM and parses the JSON
        response to extract per-criterion scores and reasoning.

        Args:
            request: Evaluation parameters including AI output and criteria.

        Returns:
            EvaluationResult with scores, composite score, and flags.
        """
        start_time = time.perf_counter()

        criteria = request.criteria if request.criteria else list(self._criteria_weights.keys())
        weights = request.criteria_weights if request.criteria_weights else self._criteria_weights

        prompt = _EVALUATION_PROMPT_TEMPLATE.format(
            criteria_list="\n".join(f"- {c}: score this criterion 0.0 to 1.0" for c in criteria),
            task_context=request.task_context[:2000],
            ai_output=request.ai_output[:4000],
        )

        logger.info(
            "Starting LLM evaluation",
            hitl_review_id=str(request.hitl_review_id),
            judge_model_id=self._judge_model_id,
            criteria=criteria,
        )

        raw_response, token_counts = await self._call_judge_llm(prompt)
        criterion_scores, reasoning = self._parse_evaluation_response(raw_response, criteria)

        composite_score = self._compute_composite_score(criterion_scores, weights)
        flagged = composite_score < self._pass_threshold

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if flagged:
            logger.warning(
                "AI output flagged for priority review",
                hitl_review_id=str(request.hitl_review_id),
                composite_score=round(composite_score, 3),
                pass_threshold=self._pass_threshold,
            )
        else:
            logger.info(
                "LLM evaluation complete — output passed",
                hitl_review_id=str(request.hitl_review_id),
                composite_score=round(composite_score, 3),
                latency_ms=latency_ms,
            )

        return EvaluationResult(
            hitl_review_id=request.hitl_review_id,
            judge_model_id=self._judge_model_id,
            criterion_scores=criterion_scores,
            composite_score=composite_score,
            judge_reasoning=reasoning,
            evaluation_latency_ms=latency_ms,
            flagged_for_review=flagged,
            evaluation_metadata={
                "prompt_length_chars": len(prompt),
                "input_tokens": token_counts.get("input_tokens", 0),
                "output_tokens": token_counts.get("output_tokens", 0),
                "criteria_evaluated": criteria,
                "weights_used": weights,
            },
        )

    async def evaluate_batch(
        self,
        requests: list[EvaluationRequest],
        max_concurrency: int = 5,
    ) -> list[EvaluationResult]:
        """Evaluate multiple AI outputs concurrently.

        Uses a semaphore to limit concurrent LLM calls and avoid rate limits.

        Args:
            requests: List of evaluation requests.
            max_concurrency: Maximum simultaneous judge LLM calls.

        Returns:
            List of EvaluationResult, one per request (in input order).
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_evaluate(req: EvaluationRequest) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(req)

        results = await asyncio.gather(
            *[_bounded_evaluate(req) for req in requests],
            return_exceptions=False,
        )
        return list(results)

    async def _call_judge_llm(
        self,
        prompt: str,
    ) -> tuple[str, dict[str, int]]:
        """Make an HTTP call to the judge LLM endpoint.

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Tuple of (response_text, token_usage_dict).
        """
        if self._client is None:
            raise RuntimeError("LLMEvaluator.initialize() was not called")

        request_body = {
            "model": self._judge_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        response = await self._client.post("/v1/messages", json=request_body)
        response.raise_for_status()
        data = response.json()

        content_blocks = data.get("content", [])
        response_text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                response_text = block.get("text", "")
                break

        usage = data.get("usage", {})
        token_counts = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        return response_text, token_counts

    def _parse_evaluation_response(
        self,
        response_text: str,
        expected_criteria: list[str],
    ) -> tuple[dict[str, float], str | None]:
        """Parse the JSON response from the judge LLM.

        Args:
            response_text: Raw text response from the judge LLM.
            expected_criteria: Criteria names that should be present.

        Returns:
            Tuple of (criterion_scores_dict, reasoning_string).
        """
        try:
            # Find JSON block in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in LLM response")

            parsed = json.loads(response_text[start:end])
            raw_scores = parsed.get("criterion_scores", {})
            reasoning = parsed.get("reasoning")

            # Clamp scores to [0.0, 1.0] and fill missing criteria with 0.5
            criterion_scores: dict[str, float] = {}
            for criterion in expected_criteria:
                raw = raw_scores.get(criterion, 0.5)
                criterion_scores[criterion] = max(0.0, min(1.0, float(raw)))

            return criterion_scores, reasoning

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse LLM evaluation response — using default scores",
                error=str(exc),
                response_preview=response_text[:200],
            )
            # Fall back to neutral scores
            return {c: 0.5 for c in expected_criteria}, None

    def _compute_composite_score(
        self,
        criterion_scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Compute the weighted composite score from criterion scores.

        Args:
            criterion_scores: Per-criterion scores (0.0-1.0).
            weights: Weight per criterion (should sum to approximately 1.0).

        Returns:
            Weighted composite score clamped to [0.0, 1.0].
        """
        if not criterion_scores:
            return 0.5

        total_weight = 0.0
        weighted_sum = 0.0

        for criterion, score in criterion_scores.items():
            weight = weights.get(criterion, 1.0 / len(criterion_scores))
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.5

        composite = weighted_sum / total_weight
        return round(max(0.0, min(1.0, composite)), 4)


__all__ = [
    "EvaluationRequest",
    "EvaluationResult",
    "LLMEvaluator",
]
