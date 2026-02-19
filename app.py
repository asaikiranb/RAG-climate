import streamlit as st
import streamlit.components.v1 as components

from retrieve import HybridRetriever
from rerank import load_reranker, rerank
from llm import get_groq_client, build_context, generate_answer
from html_renderer import build_answer_html
from query import expand_query
from config import EXAMPLE_QUERIES, RETRIEVAL_TOP_K

st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp { background-color: #ffffff; }
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    h1 { font-weight: 500; font-size: 1.6rem; color: #111; letter-spacing: -0.02em; }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #999;
        box-shadow: none;
    }
    .stButton > button {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
        color: #333;
        font-size: 13px;
        padding: 8px 14px;
        font-weight: 400;
    }
    .stButton > button:hover {
        background: #f0f0f0;
        border-color: #ccc;
    }
    .stSpinner > div { color: #666; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    return HybridRetriever()


@st.cache_resource
def load_reranker_model():
    return load_reranker()


def _render_query_input() -> str:
    """Render the text input, pre-filled from URL param if present."""
    default_query = st.query_params.get("q", "")
    return st.text_input(
        "Ask a question",
        value=default_query,
        placeholder="e.g. What is India's cooling action plan?",
        label_visibility="collapsed",
    )


def _retrieve(query: str, retriever: HybridRetriever, groq_client, reranker) -> list:
    """Expand query, run hybrid search on all variants, rerank merged pool, return top results."""
    queries = expand_query(query, groq_client)

    seen_ids = set()
    candidates = []
    for q in queries:
        for result in retriever.hybrid_search(q, reranker=reranker):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                candidates.append(result)

    return rerank(query, candidates, reranker)[:RETRIEVAL_TOP_K]


def _render_answer(query: str, retriever: HybridRetriever, groq_client, reranker):
    """Run the full pipeline and render the answer with source cards."""
    with st.spinner("Searching..."):
        try:
            results = _retrieve(query, retriever, groq_client, reranker)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.stop()

        if not results:
            st.info("No relevant documents found. Try a different query.")
            st.stop()

        context = build_context(results)
        answer = generate_answer(query, context, groq_client)

    answer_html = build_answer_html(answer, results)
    answer_lines = answer.count("\n") + 1
    estimated_height = 350 + (answer_lines * 22) + (len(results) * 55)
    estimated_height = min(max(estimated_height, 450), 1800)
    components.html(answer_html, height=estimated_height, scrolling=True)


def _render_example_queries():
    """Show example query buttons when the input is idle."""
    st.markdown("")
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 2]:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                st.query_params.update({"q": example})
                st.rerun()


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    try:
        groq_client = get_groq_client()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    try:
        with st.spinner("Loading document index..."):
            retriever = load_retriever()
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {str(e)}")
        st.stop()

    with st.spinner("Loading models..."):
        try:
            reranker = load_reranker_model()
        except Exception:
            reranker = None

    query = _render_query_input()

    if query:
        _render_answer(query, retriever, groq_client, reranker)
    else:
        _render_example_queries()


if __name__ == "__main__":
    main()
