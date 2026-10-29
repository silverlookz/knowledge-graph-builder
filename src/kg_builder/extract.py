"""Entity + relation extraction.

LLMExtractor: LangChain structured output -> list of typed triples (needs key).
RuleExtractor: deterministic pattern matcher for common business relations,
               used offline and in tests.

Both emit Triple objects with the source sentence kept as provenance, so every
edge in the graph can be traced back to the text that produced it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

ENTITY = r"((?:[A-Z][a-zA-Z&.\-]+)(?:\s+(?:of\s+)?[A-Z][a-zA-Z&.\-]+)*)"

# relation, regex, subject type, object type
PATTERNS: list[tuple[str, str, str, str]] = [
    ("ACQUIRED", rf"{ENTITY} acquired {ENTITY}", "Organization", "Organization"),
    ("HEADQUARTERED_IN", rf"{ENTITY} is headquartered in {ENTITY}", "Organization", "Location"),
    ("CEO_OF", rf"{ENTITY} is the CEO of {ENTITY}", "Person", "Organization"),
    ("FOUNDED", rf"{ENTITY} founded {ENTITY}", "Person", "Organization"),
    ("PARTNERED_WITH", rf"{ENTITY} partnered with {ENTITY}", "Organization", "Organization"),
    ("COMPETES_WITH", rf"{ENTITY} competes with {ENTITY}", "Organization", "Organization"),
    ("INVESTED_IN", rf"{ENTITY} invested in {ENTITY}", "Organization", "Organization"),
    ("SUBSIDIARY_OF", rf"{ENTITY} is a subsidiary of {ENTITY}", "Organization", "Organization"),
    ("SUPPLIES", rf"{ENTITY} supplies {ENTITY}", "Organization", "Organization"),
    ("JOINED", rf"{ENTITY} joined {ENTITY}", "Person", "Organization"),
]
LEADING_NOISE = re.compile(r"^(?:After the deal,?|Before|In \d{4},?|The)\s+", re.I)


@dataclass(frozen=True)
class Triple:
    subject: str
    subject_type: str
    relation: str
    obj: str
    object_type: str
    evidence: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _clean(name: str) -> str:
    name = LEADING_NOISE.sub("", name.strip())
    return re.sub(r"\s+(?:In|On|Before|After)$", "", name).strip(" .,")


class RuleExtractor:
    name = "rules"

    def extract(self, text: str) -> list[Triple]:
        triples = []
        for sent in split_sentences(text):
            for rel, pattern, s_type, o_type in PATTERNS:
                for m in re.finditer(pattern, sent):
                    subj, obj = _clean(m.group(1)), _clean(m.group(2))
                    if subj and obj and subj != obj:
                        triples.append(Triple(subj, s_type, rel, obj, o_type, sent))
        return triples


class LLMExtractor:  # pragma: no cover - needs network + key
    name = "llm"

    def __init__(self):
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel

        class _T(BaseModel):
            subject: str
            subject_type: str
            relation: str
            object: str
            object_type: str
            evidence: str

        class _Out(BaseModel):
            triples: list[_T]

        self.llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)\
            .with_structured_output(_Out)
        self.relations = ", ".join(p[0] for p in PATTERNS)

    def extract(self, text: str) -> list[Triple]:
        prompt = (f"Extract knowledge-graph triples. Allowed relations: {self.relations}. "
                  "Types: Person, Organization, Location. evidence = the exact sentence.\n\n" + text)
        out = self.llm.invoke(prompt)
        return [Triple(t.subject, t.subject_type, t.relation.upper(), t.object, t.object_type, t.evidence)
                for t in out.triples]


def get_extractor():
    return LLMExtractor() if os.getenv("OPENAI_API_KEY") else RuleExtractor()
