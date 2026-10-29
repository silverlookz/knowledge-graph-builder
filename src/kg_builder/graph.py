"""Knowledge graph construction, entity resolution, querying and export."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx

from .extract import Triple, get_extractor

ORG_SUFFIXES = re.compile(r"\b(inc|corp|corporation|ltd|limited|llc|co|plc|group)\.?$", re.I)
SYMMETRIC = {"PARTNERED_WITH", "COMPETES_WITH"}
TYPE_COLORS = {"Person": "#fcd34d", "Organization": "#93c5fd", "Location": "#86efac"}


def canonical_key(name: str) -> str:
    """Normalise an entity name for matching: case, punctuation, legal suffixes."""
    key = ORG_SUFFIXES.sub("", name.strip()).strip(" ,.")
    return re.sub(r"\s+", " ", key).lower()


class EntityResolver:
    """Merges surface forms that refer to the same entity.

    "Cobalt Cloud Inc." and "Cobalt Cloud" share a key; a short form like
    "Northwind" is merged into the single longer known name that starts with it.
    """

    def __init__(self):
        self.by_key: dict[str, str] = {}

    def resolve(self, name: str) -> str:
        key = canonical_key(name)
        if key in self.by_key:
            return self.by_key[key]
        prefix_matches = [v for k, v in self.by_key.items() if k.startswith(key + " ")]
        if len(prefix_matches) == 1:
            self.by_key[key] = prefix_matches[0]
            return prefix_matches[0]
        self.by_key[key] = ORG_SUFFIXES.sub("", name.strip()).strip(" ,.")
        return self.by_key[key]


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.resolver = EntityResolver()

    # ---------- build ----------
    def add_triples(self, triples: list[Triple], source: str = "corpus") -> int:
        added = 0
        for t in triples:
            s, o = self.resolver.resolve(t.subject), self.resolver.resolve(t.obj)
            self.g.add_node(s, type=self.g.nodes[s].get("type", t.subject_type) if s in self.g else t.subject_type)
            self.g.add_node(o, type=self.g.nodes[o].get("type", t.object_type) if o in self.g else t.object_type)
            if any(d["relation"] == t.relation for d in self.g.get_edge_data(s, o, default={}).values()):
                continue
            self.g.add_edge(s, o, relation=t.relation, evidence=t.evidence, source=source)
            added += 1
        return added

    @classmethod
    def from_folder(cls, folder: str | Path, extractor=None) -> "KnowledgeGraph":
        extractor = extractor or get_extractor()
        kg = cls()
        for path in sorted(Path(folder).glob("*.txt")):
            kg.add_triples(extractor.extract(path.read_text(encoding="utf-8")), source=path.name)
        return kg

    # ---------- query ----------
    def stats(self) -> dict:
        return {
            "entities": self.g.number_of_nodes(),
            "relations": self.g.number_of_edges(),
            "entity_types": dict(Counter(d["type"] for _, d in self.g.nodes(data=True))),
            "relation_types": dict(Counter(d["relation"] for *_, d in self.g.edges(data=True))),
        }

    def find(self, name: str) -> str | None:
        key = canonical_key(name)
        if key in self.resolver.by_key and self.resolver.by_key[key] in self.g:
            return self.resolver.by_key[key]
        return next((n for n in self.g if canonical_key(n) == key), None)

    def facts(self, entity: str, relation: str | None = None) -> list[dict]:
        node = self.find(entity)
        if node is None:
            return []
        out = []
        for s, o, d in list(self.g.out_edges(node, data=True)) + list(self.g.in_edges(node, data=True)):
            if relation is None or d["relation"] == relation:
                out.append({"subject": s, "relation": d["relation"], "object": o, "evidence": d["evidence"]})
        return out

    def path(self, a: str, b: str) -> list[dict]:
        """Shortest connection between two entities, ignoring edge direction."""
        na, nb = self.find(a), self.find(b)
        if not na or not nb:
            return []
        try:
            nodes = nx.shortest_path(self.g.to_undirected(as_view=True), na, nb)
        except nx.NetworkXNoPath:
            return []
        hops = []
        for u, v in zip(nodes, nodes[1:]):
            data = self.g.get_edge_data(u, v) or self.g.get_edge_data(v, u)
            d = next(iter(data.values()))
            s, o = (u, v) if self.g.has_edge(u, v) else (v, u)
            hops.append({"subject": s, "relation": d["relation"], "object": o, "evidence": d["evidence"]})
        return hops

    def central_entities(self, k: int = 5) -> list[tuple[str, float]]:
        pr = nx.pagerank(nx.DiGraph(self.g.to_undirected()))
        return [(n, round(s, 4)) for n, s in sorted(pr.items(), key=lambda kv: -kv[1])[:k]]

    def context_for(self, question: str, hops: int = 1) -> list[dict]:
        """GraphRAG-style retrieval: facts around every entity named in the question."""
        mentioned = [n for n in self.g if re.search(rf"\b{re.escape(n)}\b", question, re.I)]
        seen, facts = set(), []
        frontier = set(mentioned)
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for f in self.facts(n):
                    key = (f["subject"], f["relation"], f["object"])
                    if key not in seen:
                        seen.add(key)
                        facts.append(f)
                        nxt.update({f["subject"], f["object"]})
            frontier = nxt - frontier
        return facts

    # ---------- export ----------
    def to_json(self, path: str | Path) -> Path:
        data = nx.node_link_data(self.g, edges="links")
        Path(path).write_text(json.dumps(data, indent=2))
        return Path(path)

    def to_cypher(self, path: str | Path) -> Path:
        def q(s: str) -> str:
            return s.replace("\\", "\\\\").replace("'", "\\'")

        lines = ["// Load into Neo4j:  cat graph.cypher | cypher-shell -u neo4j -p <password>"]
        for n, d in self.g.nodes(data=True):
            lines.append(f"MERGE (:{d['type']} {{name: '{q(n)}'}});")
        for s, o, d in self.g.edges(data=True):
            lines.append(
                f"MATCH (a {{name: '{q(s)}'}}), (b {{name: '{q(o)}'}}) "
                f"MERGE (a)-[:{d['relation']} {{evidence: '{q(d['evidence'])}'}}]->(b);")
        Path(path).write_text("\n".join(lines) + "\n")
        return Path(path)

    def to_html(self, path: str | Path) -> Path:
        from pyvis.network import Network

        net = Network(height="720px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="remote")
        degree = dict(self.g.degree())
        for n, d in self.g.nodes(data=True):
            net.add_node(n, label=n, color=TYPE_COLORS.get(d["type"], "#6b7280"),
                         size=12 + 3 * degree[n], title=f"{d['type']}: {n}")
        for s, o, d in self.g.edges(data=True):
            net.add_edge(s, o, label=d["relation"].replace("_", " ").lower(), title=d["evidence"],
                         arrows="to" if d["relation"] not in SYMMETRIC else "")
        net.set_options('{"physics": {"barnesHut": {"springLength": 160}}, '
                        '"edges": {"font": {"size": 10, "align": "middle"}, "color": "#9ca3af"}}')
        net.write_html(str(path), notebook=False)
        return Path(path)

    def to_png(self, path: str | Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 9))
        pos = nx.kamada_kawai_layout(nx.Graph(self.g))
        colors = [TYPE_COLORS.get(self.g.nodes[n]["type"], "#6b7280") for n in self.g]
        degree = dict(self.g.degree())
        nx.draw_networkx_edges(self.g, pos, ax=ax, edge_color="#cbd5e1", arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.06", node_size=[300 + 120 * degree[n] for n in self.g])
        nx.draw_networkx_nodes(self.g, pos, ax=ax, node_color=colors,
                               node_size=[300 + 120 * degree[n] for n in self.g])
        nx.draw_networkx_labels(self.g, pos, ax=ax, font_size=9,
                                labels={n: n.replace(" ", "\n") for n in self.g})
        edge_labels = {(s, o): d["relation"].replace("_", " ").lower() for s, o, d in self.g.edges(data=True)}
        nx.draw_networkx_edge_labels(self.g, pos, edge_labels=edge_labels, font_size=7, font_color="#475569", ax=ax)
        for t, c in TYPE_COLORS.items():
            ax.scatter([], [], color=c, label=t, s=80)
        ax.legend(loc="lower left", frameon=False)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return Path(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    kg = KnowledgeGraph.from_folder(root / "data" / "corpus")
    out = root / "output"
    out.mkdir(exist_ok=True)
    kg.to_json(out / "graph.json"), kg.to_cypher(out / "graph.cypher"), kg.to_html(out / "graph.html")
    print(json.dumps(kg.stats(), indent=2))
    print("Most central:", kg.central_entities())
    print("Path Priya Raman -> Granite Bank:")
    for h in kg.path("Priya Raman", "Granite Bank"):
        print(f"  {h['subject']} -[{h['relation']}]-> {h['object']}")
    print(f"Exports written to {out}/")
