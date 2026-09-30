"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:         Source context (may be empty string if not applicable).
        metadata:        Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str | None = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Measure how grounded the answer is in the context."""
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        overlap = len(answer_tokens & context_tokens)
        score = overlap / len(answer_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """Measure how relevant the answer is to the question."""
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens & question_tokens)
        score = overlap / len(question_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """Measure how well the answer covers the expected answer."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens & expected_tokens)
        score = overlap / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — coverage of expected tokens across retrieved chunks."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        overlap = len(expected_tokens & union_tokens)
        score = overlap / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K)."""
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        total_relevant = 0
        precision_sum = 0.0

        for k, chunk in enumerate(contexts, start=1):
            chunk_tokens = _tokenize(chunk)
            cov = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            if cov >= relevance_threshold:
                total_relevant += 1
                precision_sum += total_relevant / k

        if total_relevant == 0:
            return 0.0
        return max(0.0, min(1.0, precision_sum / total_relevant))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """Run answer-side and optional retrieval-side evaluations."""
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5
        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        qa = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else [],
        )

        return EvalResult(
            qa_pair=qa,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query."""
    query_tokens = _tokenize(query)
    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Uses an LLM to score AI responses according to a rubric."""

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """Score an AI response using the judge LLM."""
        import json

        prompt = f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}"
        raw = self.judge_llm_fn(prompt)

        scores: dict[str, float] = {}
        reasoning = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    scores = {k: float(v) for k, v in parsed["scores"].items()}
                    reasoning = str(parsed.get("reasoning", raw))
                else:
                    scores = {
                        k: float(v)
                        for k, v in parsed.items()
                        if isinstance(v, (int, float))
                    }
                    if "reasoning" in parsed and isinstance(parsed["reasoning"], str):
                        reasoning = parsed["reasoning"]
        except Exception:
            scores = {k: 0.5 for k in rubric}
            reasoning = raw

        for k in rubric:
            if k not in scores:
                scores[k] = 0.5

        return {
            "scores": scores,
            "reasoning": reasoning,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect potential bias patterns in a batch of judge scores."""
        all_scores: list[float] = []
        first_item_avg: float | None = None

        for idx, item in enumerate(scores_batch):
            item_scores = [float(v) for v in item.get("scores", {}).values()]
            if item_scores:
                item_avg = sum(item_scores) / len(item_scores)
                if idx == 0:
                    first_item_avg = item_avg
                all_scores.extend(item_scores)

        avg_all = (sum(all_scores) / len(all_scores)) if all_scores else 0.5
        leniency_bias = avg_all > 0.8
        severity_bias = avg_all < 0.3

        positional_bias = False
        if len(scores_batch) > 1 and first_item_avg is not None:
            rest_scores: list[float] = []
            for item in scores_batch[1:]:
                rest_scores.extend([float(v) for v in item.get("scores", {}).values()])
            if rest_scores:
                rest_avg = sum(rest_scores) / len(rest_scores)
                positional_bias = (first_item_avg - rest_avg) > 0.15

        return {
            "positional_bias": bool(positional_bias),
            "leniency_bias": bool(leniency_bias),
            "severity_bias": bool(severity_bias),
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs a full evaluation benchmark."""

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """Run all QA pairs through the agent and evaluate each result."""
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            contexts_arg = pair.retrieved_contexts if pair.retrieved_contexts else None
            res = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context or "",
                expected=pair.expected_answer,
                contexts=contexts_arg,
            )
            res.qa_pair = pair
            results.append(res)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate an aggregate report from evaluation results."""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        passed_count = sum(1 for r in results if r.passed)
        avg_f = sum(r.faithfulness for r in results) / total
        avg_r = sum(r.relevance for r in results) / total
        avg_c = sum(r.completeness for r in results) / total

        recalls = [r.context_recall for r in results if r.context_recall is not None]
        avg_recall = (sum(recalls) / len(recalls)) if recalls else None

        precisions = [r.context_precision for r in results if r.context_precision is not None]
        avg_precision = (sum(precisions) / len(precisions)) if precisions else None

        failure_types: dict[str, int] = {}
        for r in results:
            if not r.passed and r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_count,
            "pass_rate": passed_count / total,
            "avg_faithfulness": avg_f,
            "avg_relevance": avg_r,
            "avg_completeness": avg_c,
            "avg_context_recall": avg_recall,
            "avg_context_precision": avg_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        """Compare new evaluation results against a baseline."""
        new_f = sum(r.faithfulness for r in new_results) / len(new_results) if new_results else 0.0
        new_r = sum(r.relevance for r in new_results) / len(new_results) if new_results else 0.0
        new_c = sum(r.completeness for r in new_results) / len(new_results) if new_results else 0.0

        base_f = sum(r.faithfulness for r in baseline_results) / len(baseline_results) if baseline_results else 0.0
        base_r = sum(r.relevance for r in baseline_results) / len(baseline_results) if baseline_results else 0.0
        base_c = sum(r.completeness for r in baseline_results) / len(baseline_results) if baseline_results else 0.0

        regressions: list[str] = []
        if (base_f - new_f) > 0.05:
            regressions.append("faithfulness")
        if (base_r - new_r) > 0.05:
            regressions.append("relevance")
        if (base_c - new_c) > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """Return EvalResults where any score is below threshold."""
        return [
            r for r in results
            if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Analyzes failed evaluation results to identify patterns and suggest fixes."""

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """Count failures by failure_type."""
        counts: dict[str, int] = {}
        for f in failures:
            ftype = f.failure_type or "unknown"
            counts[ftype] = counts.get(ftype, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """Suggest a root cause for a single failure based on its scores."""
        f = failure.faithfulness
        r = failure.relevance
        c = failure.completeness
        min_val = min(f, r, c)

        low_count = sum(1 for val in (f, r, c) if abs(val - min_val) < 1e-6)
        if low_count > 1 or (f < 0.3 and r < 0.3):
            return "Multiple issues detected — review full pipeline"
        if min_val == f:
            return "Context is missing or irrelevant — improve retrieval"
        elif min_val == r:
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """Generate a prioritized list of improvement suggestions based on failure patterns."""
        if not failures:
            return []

        suggestions: list[str] = []
        categories = self.categorize_failures(failures)

        if "hallucination" in categories or any(f.faithfulness < 0.5 for f in failures):
            suggestions.append("Implement hallucination checker and ground responses strictly in retrieved context")
        if "irrelevant" in categories or "off_topic" in categories or any(f.relevance < 0.5 for f in failures):
            suggestions.append("Refine system prompts and query rewriting to keep answers tightly focused on user questions")
        if "incomplete" in categories or any(f.completeness < 0.5 for f in failures):
            suggestions.append("Increase chunk size or top_k in RAG pipeline to capture missing context")

        default_suggestions = [
            "Improve retrieval quality by fine-tuning chunking strategy and BM25/embedding parameters",
            "Add few-shot domain examples to prompt template for standardizing response completeness",
            "Deploy a verification judge step in CI/CD pipeline before pushing prompt changes",
        ]
        for ds in default_suggestions:
            if len(suggestions) < 3 and ds not in suggestions:
                suggestions.append(ds)

        return suggestions

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions."""
        rows = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for i, f in enumerate(failures, start=1):
            fid = f"F{i:03d}"
            ftype = f.failure_type or "Unknown"
            rc = self.find_root_cause(f)
            sug = (
                suggestions[i - 1]
                if i - 1 < len(suggestions)
                else (suggestions[0] if suggestions else "Review and iterate")
            )
            rows.append(f"| {fid} | {ftype} | {rc} | {sug} | Open |")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
