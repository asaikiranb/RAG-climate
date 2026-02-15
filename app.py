import streamlit as st
import streamlit.components.v1 as components
import os
import re
import html as html_lib
from dotenv import load_dotenv

from src.config import get_settings
from src.retrieval import HybridRetriever
from src.generation import AnswerGenerator

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="auto"
)

# Clean minimal styling
st.markdown("""
<style>
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

# System prompt — concise, no bold
SYSTEM_PROMPT = """You are a research assistant. Answer the question using ONLY the provided sources.

RULES:
1. Be concise. Get to the point. No filler, no restating the question. Aim for 100-200 words.
2. Cite inline. After a key claim, add the source number: "India joined in 1992 [1]." Use only ONE citation per claim.
3. Use bullet points for lists. Do NOT use bold or any special formatting.
4. Be specific. Include dates, numbers, names from the sources.
5. Only cite specific facts. Connecting sentences don't need citations.
6. If unsure, say "The documents don't cover this."
7. Do NOT use markdown headers or bold text. Write in plain text only.

Sources:
{context}

Question: {query}

Answer:"""


@st.cache_resource
def load_system():
    """Load settings, retriever, and generator."""
    settings = get_settings()
    retriever = HybridRetriever(settings)
    generator = AnswerGenerator(settings)
    return settings, retriever, generator


def build_answer_html(answer_text, results):
    """
    Build a self-contained HTML block with the answer, inline clickable citations,
    and collapsible source cards.
    """
    # Build source data
    sources = []
    for i, result in enumerate(results, 1):
        meta = result.get('metadata', {})
        # Handle both old and new metadata formats
        filename = meta.get('filename', 'Unknown')
        page_number = meta.get('page_number', '?')
        score = meta.get('score', result.get('score', 0.0))

        display_name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        document_text = result.get('document', result.get('text', ''))

        sources.append({
            'num': i,
            'filename': filename,
            'display_name': display_name,
            'page': page_number,
            'score': score,
            'text': html_lib.escape(document_text)
        })

    # Escape the answer text for HTML
    safe_answer = html_lib.escape(answer_text)

    # Strip any bold markers the LLM might still produce
    safe_answer = re.sub(r'\*\*(.+?)\*\*', r'\1', safe_answer)

    # Convert bullet points
    safe_answer = re.sub(r'^- (.+)$', r'<li>\1</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'^(\* )(.+)$', r'<li>\2</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', safe_answer)

    # Convert numbered lists
    safe_answer = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', safe_answer, flags=re.MULTILINE)

    # Replace [N] with clickable citation pills
    def replace_citation(match):
        num = match.group(1)
        return f'<span class="cite" onclick="showSource({num})">{num}</span>'

    safe_answer = re.sub(r'\[(\d+)\]', replace_citation, safe_answer)

    # Convert newlines to paragraphs
    paragraphs = safe_answer.split('\n')
    formatted = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<ul>') and not p.startswith('<li>') and not p.startswith('</ul>'):
            formatted.append(f'<p>{p}</p>')
        elif p:
            formatted.append(p)
    safe_answer = '\n'.join(formatted)

    # Build source cards
    source_cards_html = ""
    for src in sources:
        source_cards_html += f"""
        <div class="source-card" id="source-{src['num']}">
            <div class="source-header" onclick="toggleSource({src['num']})">
                <div class="source-num">{src['num']}</div>
                <div class="source-meta">
                    <div class="source-title">{src['display_name']}</div>
                    <div class="source-page">Page {src['page']}</div>
                </div>
                <div class="source-chevron" id="chevron-{src['num']}">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            <div class="source-body" id="body-{src['num']}">
                <div class="source-text">{src['text']}</div>
                <div class="source-file">{src['filename']}</div>
            </div>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            color: #222;
            line-height: 1.7;
            font-size: 14.5px;
            background: transparent;
            -webkit-font-smoothing: antialiased;
        }}

        .answer-content {{
            padding: 0 0 20px 0;
        }}
        .answer-content p {{
            margin-bottom: 8px;
        }}
        .answer-content ul {{
            margin: 6px 0 10px 18px;
            padding: 0;
        }}
        .answer-content li {{
            margin-bottom: 5px;
            line-height: 1.6;
        }}

        .cite {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 600;
            min-width: 15px;
            height: 15px;
            padding: 0 3px;
            border-radius: 3px;
            background: #eee;
            color: #666;
            cursor: pointer;
            vertical-align: super;
            margin: 0 1px;
            line-height: 1;
            transition: all 0.15s ease;
            position: relative;
            top: -1px;
        }}
        .cite:hover {{
            background: #ddd;
            color: #333;
        }}
        .cite.active {{
            background: #333;
            color: #fff;
        }}

        .sources-section {{
            border-top: 1px solid #eee;
            padding-top: 16px;
            margin-top: 4px;
        }}
        .sources-label {{
            font-size: 11px;
            font-weight: 500;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
        }}

        .source-card {{
            border: 1px solid #eee;
            border-radius: 8px;
            margin-bottom: 6px;
            overflow: hidden;
            transition: border-color 0.2s ease;
            background: #fff;
        }}
        .source-card:hover {{
            border-color: #ccc;
        }}
        .source-card.highlighted {{
            border-color: #333;
        }}

        .source-header {{
            display: flex;
            align-items: center;
            padding: 10px 14px;
            cursor: pointer;
            user-select: none;
            gap: 10px;
        }}
        .source-header:hover {{
            background: #fafafa;
        }}

        .source-num {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 4px;
            background: #f5f5f5;
            color: #666;
            font-size: 11px;
            font-weight: 500;
            flex-shrink: 0;
        }}

        .source-meta {{
            flex: 1;
            min-width: 0;
        }}
        .source-title {{
            font-size: 13px;
            font-weight: 400;
            color: #333;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .source-page {{
            font-size: 11px;
            color: #999;
            margin-top: 1px;
        }}

        .source-chevron {{
            color: #999;
            transition: transform 0.2s ease;
            flex-shrink: 0;
        }}
        .source-chevron.open {{
            transform: rotate(180deg);
        }}

        .source-body {{
            display: none;
            padding: 0 14px 12px 44px;
        }}
        .source-body.open {{
            display: block;
        }}
        .source-text {{
            font-size: 12.5px;
            line-height: 1.55;
            color: #555;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #f0f0f0;
        }}
        .source-file {{
            font-size: 11px;
            color: #aaa;
            margin-top: 6px;
        }}
    </style>
    </head>
    <body>
        <div class="answer-content">
            {safe_answer}
        </div>

        <div class="sources-section">
            <div class="sources-label">Sources</div>
            {source_cards_html}
        </div>

        <script>
            function toggleSource(num) {{
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                const isOpen = body.classList.contains('open');
                if (isOpen) {{
                    body.classList.remove('open');
                    chevron.classList.remove('open');
                }} else {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}
            }}

            function showSource(num) {{
                const card = document.getElementById('source-' + num);
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                if (!card) return;

                document.querySelectorAll('.source-card.highlighted').forEach(el => {{
                    el.classList.remove('highlighted');
                }});
                document.querySelectorAll('.cite.active').forEach(el => {{
                    el.classList.remove('active');
                }});

                event.target.classList.add('active');

                if (!body.classList.contains('open')) {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}

                card.classList.add('highlighted');
                card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                setTimeout(() => {{
                    card.classList.remove('highlighted');
                }}, 2500);
            }}
        </script>
    </body>
    </html>
    """
    return full_html


def generate_answer_simple(query: str, context_chunks: list, generator: AnswerGenerator) -> tuple:
    """
    Generate answer using new modular architecture.
    Returns tuple of (answer_text, generated_answer_object).
    """
    try:
        generated = generator.generate(
            query=query,
            context_chunks=context_chunks,
            enable_validation=st.session_state.get('enable_validation', False)
        )
        return generated.answer, generated
    except Exception as e:
        return f"Error generating answer: {str(e)}", None


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    # Sidebar for advanced features
    with st.sidebar:
        st.header("Settings")

        enable_advanced = st.checkbox(
            "Enable Advanced Features",
            value=True,
            help="Enable HyDE, query expansion, reranking, and other advanced retrieval features"
        )

        show_metadata = st.checkbox(
            "Show Retrieval Metadata",
            value=False,
            help="Display confidence scores and retrieval details"
        )

        enable_validation = st.checkbox(
            "Enable Answer Validation",
            value=False,
            help="Validate citations and answer quality (slower but more accurate)"
        )

        st.session_state['enable_validation'] = enable_validation
        st.session_state['show_metadata'] = show_metadata

        st.divider()
        st.caption("Using modular architecture with HybridRetriever and AnswerGenerator")

    # Load system components
    try:
        with st.spinner("Loading system..."):
            settings, retriever, generator = load_system()

            # Update settings based on sidebar toggles
            if not enable_advanced:
                settings.retrieval.use_hyde = False
                settings.retrieval.use_query_expansion = False
                settings.retrieval.use_reranking = False
                settings.retrieval.use_mmr = False

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

    # Read query from URL params (set by example buttons)
    default_query = st.query_params.get("q", "")

    # Search input
    query = st.text_input(
        "Ask a question",
        value=default_query,
        placeholder="e.g. What is India's cooling action plan?",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Searching..."):
            try:
                # Use new modular retriever
                results = retriever.search(
                    query=query,
                    top_k=5,
                    return_metadata=True
                )
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.stop()

            if not results:
                st.info("No relevant documents found. Try a different query.")
                st.stop()

            # Show metadata if enabled
            if st.session_state.get('show_metadata', False):
                with st.expander("Retrieval Metadata", expanded=False):
                    if results and 'retrieval_metadata' in results[0]:
                        meta = results[0]['retrieval_metadata']
                        st.json(meta)

            # Generate answer using new generator
            answer, generated = generate_answer_simple(query, results, generator)

        # Show confidence and validation scores if enabled
        if st.session_state.get('show_metadata', False) and generated:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{generated.confidence_score:.2%}")
            with col2:
                if generated.citation_validation:
                    st.metric("Citation Score", f"{generated.citation_validation.citation_score:.2%}")
            with col3:
                if generated.answer_verification:
                    st.metric("Answer Quality", f"{generated.answer_verification.overall_score:.2%}")

        # Render answer with citations
        answer_html = build_answer_html(answer, results)

        answer_lines = answer.count('\n') + 1
        estimated_height = 350 + (answer_lines * 22) + (len(results) * 55)
        estimated_height = min(max(estimated_height, 450), 1800)

        components.html(answer_html, height=estimated_height, scrolling=True)

    # Example queries when no question asked
    if not query:
        st.markdown("")
        st.markdown("##### Try asking")
        example_queries = [
            "What is the Montreal Protocol and India's role in it?",
            "What are low-GWP refrigerant alternatives?",
            "What are passive cooling strategies for buildings?",
            "What training is required for RAC technicians?",
            "What is the India Cooling Action Plan?",
        ]

        cols = st.columns(2)
        for idx, example in enumerate(example_queries):
            with cols[idx % 2]:
                if st.button(example, key=f"ex_{idx}", use_container_width=True):
                    st.query_params.update({"q": example})
                    st.rerun()


if __name__ == "__main__":
    main()
