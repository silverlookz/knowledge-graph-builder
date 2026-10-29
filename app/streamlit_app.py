"""Streamlit explorer.  Run: streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kg_builder.extract import get_extractor  # noqa: E402
from kg_builder.graph import KnowledgeGraph  # noqa: E402
from kg_builder.qa import answer  # noqa: E402

st.set_page_config(page_title="Knowledge Graph Builder", page_icon="🕸️", layout="wide")
st.title("🕸️ Knowledge Graph Builder")

text = st.text_area("Source text (edit or paste your own)",
                    (ROOT / "data/corpus/news_briefs.txt").read_text(), height=180)
kg = KnowledgeGraph()
kg.add_triples(get_extractor().extract(text))
s = kg.stats()
c1, c2, c3 = st.columns(3)
c1.metric("Entities", s["entities"]); c2.metric("Relations", s["relations"])
c3.metric("Relation types", len(s["relation_types"]))

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    kg.to_html(f.name)
    components.html(Path(f.name).read_text(), height=740)

left, right = st.columns(2)
with left:
    st.subheader("Ask the graph")
    q = st.text_input("Question", "Who is the CEO of the company that acquired Larkspur Analytics?")
    if q:
        r = answer(kg, q)
        st.write(r["answer"])
        st.dataframe(r["facts"], use_container_width=True)
with right:
    st.subheader("How are two entities connected?")
    names = sorted(kg.g.nodes)
    if len(names) >= 2:
        a = st.selectbox("From", names, index=0)
        b = st.selectbox("To", names, index=len(names) - 1)
        for h in kg.path(a, b):
            st.markdown(f"**{h['subject']}** → `{h['relation']}` → **{h['object']}**  \n_{h['evidence']}_")
