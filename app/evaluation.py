"""Part 1 evaluation: same five questions, three techniques, scored 1 to 5.

Scoring uses an "LLM as judge": a second model call reads the question, the
retrieved context, an optional reference answer and the candidate answer, then
returns scores for accuracy, clarity and relevance. The judge runs at
temperature 0 so the same answer always gets the same score.
"""
import json
from datetime import datetime, timezone

from .llm import LLMError
from .prompts import TECHNIQUES, format_context
from .rag import load_json_list

METRICS = ("accuracy", "clarity", "relevance")

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of question-answering systems. "
    "Reply with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """Evaluate the candidate answer.

Question: {question}

Retrieved context:
{context}

Reference answer (may be empty): {reference}

Candidate answer:
{answer}

Score each criterion from 1 (poor) to 5 (excellent):
- accuracy: facts are correct and supported by the context or reference
- clarity: easy to read, well organised, no padding
- relevance: answers the question directly with nothing unrelated

Reply as JSON: {{"accuracy": <int>, "clarity": <int>, "relevance": <int>, "reason": "<one short sentence>"}}"""


def _judge(llm, model, question, chunks, reference, answer) -> dict:
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context=format_context(chunks),
        reference=reference or "(none)",
        answer=answer,
    )
    raw = llm.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        model=model, temperature=0, max_tokens=200, json_mode=True,
    )
    data = json.loads(raw)
    scores = {m: max(1, min(5, int(data[m]))) for m in METRICS}
    return {"scores": scores, "reason": str(data.get("reason", "")).strip()}


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarise(questions: list[dict]) -> dict:
    summary = {}
    for technique in TECHNIQUES:
        per_metric = {m: [] for m in METRICS}
        for q in questions:
            scores = q["results"][technique].get("scores")
            if scores:
                for m in METRICS:
                    per_metric[m].append(scores[m])
        row = {m: _average(per_metric[m]) for m in METRICS}
        valid = [v for v in row.values() if v is not None]
        row["overall"] = _average(valid)
        row["scored_questions"] = len(per_metric["accuracy"])
        summary[technique] = row
    return summary


def _analysis(llm, summary, questions, winner) -> str:
    notes = []
    for q in questions:
        for technique, result in q["results"].items():
            if result.get("reason"):
                notes.append(f"- {TECHNIQUES[technique]['label']}: {result['reason']}")
    prompt = (
        "Average scores (1-5) for three prompting techniques on the same questions:\n"
        f"{json.dumps(summary, indent=2)}\n\nJudge comments:\n" + "\n".join(notes[:15]) +
        f"\n\nThe highest overall score belongs to {TECHNIQUES[winner]['label']}. "
        "In 4 to 5 plain sentences for a student presentation, explain which technique "
        "performed best and why, mentioning accuracy, clarity and relevance. "
        "Base it only on the numbers and comments above."
    )
    return llm.chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300)


def run_evaluation(rag, judge_model: str, questions_file, results_file) -> dict:
    items = load_json_list(questions_file)
    if not items:
        raise ValueError(f"No evaluation questions found in {questions_file}")

    questions = []
    for item in items:
        question, reference = item["question"], item.get("reference", "")
        chunks = rag.retrieve(question)  # retrieved once, shared by all three techniques
        results = {}
        for technique in TECHNIQUES:
            entry = {"answer": "", "scores": None, "reason": "", "error": None}
            try:
                entry["answer"] = rag.generate(technique, question, chunks)["answer"]
                entry.update(_judge(rag.llm, judge_model, question, chunks, reference, entry["answer"]))
            except (LLMError, ValueError, KeyError, TypeError) as exc:
                entry["error"] = str(exc)
            results[technique] = entry
        questions.append({
            "question": question,
            "reference": reference,
            "sources": sorted({c["source"] for c in chunks}),
            "results": results,
        })

    summary = summarise(questions)
    scored = {t: s["overall"] for t, s in summary.items() if s["overall"] is not None}
    winner = max(scored, key=scored.get) if scored else None

    analysis = ""
    if winner:
        try:
            analysis = _analysis(rag.llm, summary, questions, winner)
        except LLMError as exc:
            analysis = f"Analysis unavailable: {exc}"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": rag.llm.model,
        "judge_model": judge_model,
        "questions": questions,
        "summary": summary,
        "winner": winner,
        "analysis": analysis,
    }
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_latest(results_file) -> dict | None:
    if not results_file.exists():
        return None
    try:
        return json.loads(results_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
