import json

import pytest

from kg_builder.extract import RuleExtractor
from kg_builder.graph import EntityResolver, KnowledgeGraph
from kg_builder.qa import answer


@pytest.fixture(scope="module")
def kg():
    return KnowledgeGraph.from_folder("data/corpus", extractor=RuleExtractor())


def test_extraction_types():
    t = RuleExtractor().extract("Ada Lovelace is the CEO of Engine Works. Engine Works is headquartered in London.")
    assert [(x.subject, x.relation, x.obj) for x in t] == [
        ("Ada Lovelace", "CEO_OF", "Engine Works"), ("Engine Works", "HEADQUARTERED_IN", "London")]
    assert t[0].subject_type == "Person" and t[1].object_type == "Location"


def test_entity_resolution():
    r = EntityResolver()
    assert r.resolve("Cobalt Cloud Inc.") == "Cobalt Cloud"
    assert r.resolve("cobalt cloud") == "Cobalt Cloud"
    assert r.resolve("Cobalt") == "Cobalt Cloud"


def test_graph_shape(kg):
    s = kg.stats()
    assert s["entities"] == 20 and s["relations"] == 28
    assert s["entity_types"] == {"Organization": 8, "Location": 6, "Person": 6}


def test_no_duplicate_edges():
    kg = KnowledgeGraph()
    triples = RuleExtractor().extract("Acme Corp acquired Beta Labs. Acme acquired Beta Labs.")
    kg.add_triples(triples)
    assert kg.g.number_of_edges() == 1


def test_facts_and_evidence(kg):
    facts = kg.facts("Maya Chen", relation="FOUNDED")
    assert facts[0]["object"] == "Quill Robotics"
    assert "Maya Chen founded Quill Robotics" in facts[0]["evidence"]


def test_path(kg):
    hops = kg.path("Priya Raman", "Granite Bank")
    assert hops[0]["subject"] == "Priya Raman" and len(hops) <= 4


def test_multi_hop_qa(kg):
    r = answer(kg, "Who is the CEO of the company that acquired Larkspur Analytics?")
    pairs = {(f["subject"], f["relation"]) for f in r["facts"]}
    assert ("Daniel Okafor", "CEO_OF") in pairs


def test_exports(kg, tmp_path):
    data = json.loads(kg.to_json(tmp_path / "g.json").read_text())
    assert len(data["nodes"]) == 20
    cypher = kg.to_cypher(tmp_path / "g.cypher").read_text()
    assert cypher.count("MERGE (:") == 20 and ":ACQUIRED" in cypher
    assert "vis-network" in kg.to_html(tmp_path / "g.html").read_text()
