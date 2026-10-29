"""GraphRAG question answering.

Finds entities named in the question, pulls their 2-hop neighbourhood from the
graph, and either (a) asks an LLM to answer from those facts only, or
(b) offline, returns the facts ranked by overlap with the question.
Every fact carries its evidence sentence, so answers stay traceable.
"""
from __future__ import annotations

import os
import re

from .graph import KnowledgeGraph

REL_HINTS = {
    "CEO_OF": ["ceo", "runs", "leads", "head", "chief executive"],
    "ACQUIRED": ["acquired", "bought", "acquisition", "owns"],
    "HEADQUARTERED_IN": ["headquartered", "based", "located", "where"],
    "FOUNDED": ["founded", "founder", "started"],
    "PARTNERED_WITH": ["partner", "partnered", "partnership"],
    "COMPETES_WITH": ["compete", "competitor", "rival"],
    "INVESTED_IN": ["invest", "investor", "backed"],
    "SUPPLIES": ["supplies", "supplier", "vendor", "provides"],
    "SUBSIDIARY_OF": ["subsidiary", "parent"],
    "JOINED": ["joined", "works", "hired"],
}


def _score(question: str, fact: dict) -> int:
    q = question.lower()
    score = sum(2 for w in REL_HINTS.get(fact["relation"], []) if re.search(rf"\b{w}", q))
    score += sum(1 for n in (fact["subject"], fact["object"]) if n.lower() in q)
    return score


def answer(kg: KnowledgeGraph, question: str, hops: int = 2, k: int = 6) -> dict:
    facts = kg.context_for(question, hops=hops)
    if not facts:
        return {"answer": "No entities from the question were found in the graph.", "facts": [], "mode": "none"}
    ranked = sorted(facts, key=lambda f: -_score(question, f))[:k]

    if os.getenv("OPENAI_API_KEY"):  # pragma: no cover
        from langchain_openai import ChatOpenAI

        lines = "\n".join(f"- {f['subject']} {f['relation']} {f['object']}" for f in facts)
        llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)
        text = llm.invoke(f"Answer using ONLY these graph facts. If they are insufficient, say so.\n"
                          f"{lines}\n\nQuestion: {question}").content.strip()
        return {"answer": text, "facts": ranked, "mode": "llm"}

    summary = "; ".join(f"{f['subject']} {f['relation'].replace('_', ' ').lower()} {f['object']}"
                        for f in ranked[:3])
    return {"answer": f"Most relevant facts: {summary}.", "facts": ranked, "mode": "graph"}
