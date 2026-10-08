"""Part 1: the three prompt templates.

All three receive the SAME retrieved context and the SAME question. Only the
way we ask changes, so any difference in the answers is caused by the prompt.

    zero_shot   plain instruction, no examples
    few_shot    the same instruction plus worked examples (loaded from JSON)
    role_based  a system message that gives the model a job and rules
"""

TECHNIQUES = {
    "zero_shot": {
        "label": "Zero-shot",
        "description": "Just the instruction. No examples, no persona.",
    },
    "few_shot": {
        "label": "Few-shot",
        "description": "The instruction plus worked examples to copy the style from.",
    },
    "role_based": {
        "label": "Role-based",
        "description": "The model is told to act as a technical document analyst with rules.",
    },
}

ZERO_SHOT_TEMPLATE = """Answer the question using the context below.

Context:
{context}

Question: {question}

Answer:"""

FEW_SHOT_TEMPLATE = """Answer the question using only the context. Follow the style of the examples.

{examples}

Now answer the real question.

Context:
{context}

Question: {question}

Answer:"""

ROLE_SYSTEM_PROMPT = """You are a senior technical document analyst. You read excerpts from technical documents and answer questions about them precisely.

Rules:
1. Use only the information in the numbered excerpts. Never use outside knowledge.
2. If the excerpts do not contain the answer, reply exactly: "The provided documents do not contain this information."
3. Be concise: two to four sentences, or a short list when the answer is a set of items.
4. Cite the excerpts you used with their numbers, for example [1] or [2]."""

ROLE_USER_TEMPLATE = """Excerpts:
{context}

Question: {question}"""

EXAMPLE_TEMPLATE = """Example {number}
Context:
{context}
Question: {question}
Answer: {answer}"""


def format_context(chunks: list[dict]) -> str:
    """Number the retrieved chunks so the model can cite them as [1], [2], [3]."""
    if not chunks:
        return "(no context was retrieved)"
    return "\n\n".join(
        f"[{i}] (source: {chunk['source']})\n{chunk['text']}" for i, chunk in enumerate(chunks, 1)
    )


def format_examples(examples: list[dict]) -> str:
    return "\n\n".join(
        EXAMPLE_TEMPLATE.format(
            number=i, context=ex["context"], question=ex["question"], answer=ex["answer"]
        )
        for i, ex in enumerate(examples, 1)
    )


def build_messages(technique: str, question: str, chunks: list[dict], examples: list[dict]):
    """Return the chat messages for the chosen technique."""
    if technique not in TECHNIQUES:
        raise ValueError(f"Unknown technique '{technique}'")
    context = format_context(chunks)

    if technique == "zero_shot":
        return [{"role": "user", "content": ZERO_SHOT_TEMPLATE.format(context=context, question=question)}]

    if technique == "few_shot":
        body = FEW_SHOT_TEMPLATE.format(
            examples=format_examples(examples), context=context, question=question
        )
        return [{"role": "user", "content": body}]

    return [
        {"role": "system", "content": ROLE_SYSTEM_PROMPT},
        {"role": "user", "content": ROLE_USER_TEMPLATE.format(context=context, question=question)},
    ]


def render_prompt(messages: list[dict]) -> str:
    """Human-readable version of the messages, shown in the UI for transparency."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
