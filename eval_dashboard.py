"""
Streamlit Evaluation Dashboard
Separate app for visualizing RAG system performance metrics.

Run with: streamlit run eval_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.evaluation import RAGEvaluator, DatasetGenerator
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .stMetric { background: white; padding: 1rem; border-radius: 0.5rem; }
    h1 { color: #1f2937; }
    h2 { color: #374151; margin-top: 2rem; }
    h3 { color: #4b5563; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_evaluator():
    """Load evaluator (cached)."""
    settings = get_settings()
    return RAGEvaluator(settings)


@st.cache_resource
def load_dataset_generator():
    """Load dataset generator (cached)."""
    settings = get_settings()
    return DatasetGenerator(settings)


def load_test_dataset(file_path: str) -> List[Dict]:
    """Load test dataset from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, filename: str):
    """Save evaluation results to file."""
    output_dir = Path("evals/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return output_file


def plot_retrieval_metrics(metrics: Dict) -> go.Figure:
    """Create visualization for retrieval metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate@K'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    k_values = [1, 3, 5, 10]

    # Precision@K
    precision_values = [metrics.get(f'precision_at_{k}', 0) for k in k_values]
    fig.add_trace(
        go.Bar(x=k_values, y=precision_values, name='Precision', marker_color='#3b82f6'),
        row=1, col=1
    )

    # Recall@K
    recall_values = [metrics.get(f'recall_at_{k}', 0) for k in k_values]
    fig.add_trace(
        go.Bar(x=k_values, y=recall_values, name='Recall', marker_color='#10b981'),
        row=1, col=2
    )

    # NDCG@K
    ndcg_values = [metrics.get(f'ndcg_at_{k}', 0) for k in k_values]
    fig.add_trace(
        go.Bar(x=k_values, y=ndcg_values, name='NDCG', marker_color='#f59e0b'),
        row=2, col=1
    )

    # Hit Rate@K
    hitrate_values = [metrics.get(f'hit_rate_at_{k}', 0) for k in k_values]
    fig.add_trace(
        go.Bar(x=k_values, y=hitrate_values, name='Hit Rate', marker_color='#8b5cf6'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False, title_text="Retrieval Performance Metrics")
    fig.update_xaxes(title_text="K", row=2, col=1)
    fig.update_xaxes(title_text="K", row=2, col=2)
    fig.update_yaxes(title_text="Score", range=[0, 1])

    return fig


def plot_generation_metrics(metrics: Dict) -> go.Figure:
    """Create radar chart for generation metrics."""
    categories = ['Faithfulness', 'Relevance', 'Citation Accuracy', 'Completeness', 'Clarity']
    values = [
        metrics.get('faithfulness', 0),
        metrics.get('relevance', 0),
        metrics.get('citation_accuracy', 0),
        metrics.get('completeness', 0),
        metrics.get('clarity', 0.8)  # Default for clarity
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Generation Quality',
        marker_color='#3b82f6'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Generation Quality Metrics",
        height=500
    )

    return fig


def plot_ablation_comparison(ablation_results: List[Dict]) -> go.Figure:
    """Create comparison chart for ablation study."""
    df = pd.DataFrame(ablation_results)

    fig = go.Figure()

    metrics_to_plot = ['precision_at_5', 'recall_at_5', 'mrr', 'faithfulness', 'relevance']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

    for metric, color in zip(metrics_to_plot, colors):
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=df['config_name'],
                y=df[metric],
                marker_color=color
            ))

    fig.update_layout(
        barmode='group',
        title="Ablation Study: Configuration Comparison",
        xaxis_title="Configuration",
        yaxis_title="Score",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def main():
    st.title("📊 RAG System Evaluation Dashboard")
    st.markdown("Comprehensive performance analysis and metrics visualization")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    eval_mode = st.sidebar.selectbox(
        "Evaluation Mode",
        ["Quick Eval", "Full Evaluation", "Ablation Study", "Dataset Generation"]
    )

    # Load evaluator
    try:
        evaluator = load_evaluator()
        st.sidebar.success("✅ Evaluator loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load evaluator: {e}")
        st.error("Please ensure your .env file is configured correctly")
        return

    # Main content based on mode
    if eval_mode == "Quick Eval":
        st.header("🚀 Quick Evaluation")

        col1, col2 = st.columns([2, 1])

        with col1:
            test_file = st.selectbox(
                "Select Test Dataset",
                ["evals/datasets/example_test_set.json"] +
                [str(p) for p in Path("evals/datasets").glob("*.json") if p.name != "ablation_configs.json"]
            )

        with col2:
            top_k = st.slider("Top K Results", 1, 20, 5)

        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    # Load dataset
                    test_data = load_test_dataset(test_file)
                    st.info(f"Loaded {len(test_data)} test examples")

                    # Run evaluation
                    result = evaluator.evaluate_complete(test_data, top_k=top_k)

                    # Display results
                    st.success("✅ Evaluation complete!")

                    # Summary metrics
                    st.subheader("📈 Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Precision@5", f"{result.retrieval_metrics.precision_at_5:.2%}")
                    with col2:
                        st.metric("Recall@5", f"{result.retrieval_metrics.recall_at_5:.2%}")
                    with col3:
                        st.metric("Faithfulness", f"{result.generation_metrics.faithfulness:.2%}")
                    with col4:
                        st.metric("Avg Latency", f"{result.e2e_metrics.average_latency:.2f}s")

                    # Detailed metrics
                    tab1, tab2, tab3, tab4 = st.tabs(["📍 Retrieval", "✍️ Generation", "⚡ Performance", "📝 Details"])

                    with tab1:
                        st.plotly_chart(
                            plot_retrieval_metrics(result.retrieval_metrics.__dict__),
                            use_container_width=True
                        )

                        # Retrieval metrics table
                        st.subheader("Detailed Retrieval Metrics")
                        retrieval_df = pd.DataFrame({
                            'Metric': ['MRR', 'MAP', 'NDCG@5', 'Hit Rate@5'],
                            'Value': [
                                f"{result.retrieval_metrics.mrr:.3f}",
                                f"{result.retrieval_metrics.map:.3f}",
                                f"{result.retrieval_metrics.ndcg_at_5:.3f}",
                                f"{result.retrieval_metrics.hit_rate_at_5:.3f}"
                            ]
                        })
                        st.dataframe(retrieval_df, use_container_width=True)

                    with tab2:
                        st.plotly_chart(
                            plot_generation_metrics(result.generation_metrics.__dict__),
                            use_container_width=True
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Citation Accuracy", f"{result.generation_metrics.citation_accuracy:.2%}")
                            st.metric("Completeness", f"{result.generation_metrics.completeness:.2%}")
                        with col2:
                            if result.generation_metrics.bleu_score:
                                st.metric("BLEU Score", f"{result.generation_metrics.bleu_score:.3f}")

                    with tab3:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieval Time", f"{result.e2e_metrics.retrieval_time:.2f}s")
                        with col2:
                            st.metric("Generation Time", f"{result.e2e_metrics.generation_time:.2f}s")
                        with col3:
                            st.metric("Total Time", f"{result.e2e_metrics.average_latency:.2f}s")

                        # Performance breakdown
                        perf_df = pd.DataFrame({
                            'Component': ['Retrieval', 'Generation'],
                            'Time (s)': [
                                result.e2e_metrics.retrieval_time,
                                result.e2e_metrics.generation_time
                            ]
                        })
                        fig = px.pie(perf_df, values='Time (s)', names='Component',
                                   title='Time Distribution')
                        st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        st.subheader("Full Results")
                        st.json(result.to_dict())

                        # Save results
                        if st.button("💾 Save Results"):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"eval_results_{timestamp}.json"
                            output_file = save_results(result.to_dict(), filename)
                            st.success(f"Results saved to {output_file}")

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logger.exception("Evaluation error")

    elif eval_mode == "Full Evaluation":
        st.header("🔬 Comprehensive Evaluation")
        st.info("This runs a complete evaluation including RAGAS metrics (if available)")

        test_file = st.selectbox(
            "Select Test Dataset",
            ["evals/datasets/example_test_set.json"] +
            [str(p) for p in Path("evals/datasets").glob("*.json")]
        )

        include_ragas = st.checkbox("Include RAGAS Metrics", value=True)

        if st.button("Run Full Evaluation", type="primary"):
            with st.spinner("Running comprehensive evaluation..."):
                test_data = load_test_dataset(test_file)

                result = evaluator.evaluate_complete(
                    test_data,
                    compute_ragas=include_ragas
                )

                st.success("✅ Complete evaluation finished!")

                # Display comprehensive results
                st.subheader("📊 All Metrics")

                # Create metrics summary
                metrics_data = []

                # Retrieval
                metrics_data.append({
                    'Category': 'Retrieval',
                    'Metric': 'Precision@5',
                    'Value': f"{result.retrieval_metrics.precision_at_5:.2%}"
                })
                metrics_data.append({
                    'Category': 'Retrieval',
                    'Metric': 'MRR',
                    'Value': f"{result.retrieval_metrics.mrr:.3f}"
                })

                # Generation
                metrics_data.append({
                    'Category': 'Generation',
                    'Metric': 'Faithfulness',
                    'Value': f"{result.generation_metrics.faithfulness:.2%}"
                })
                metrics_data.append({
                    'Category': 'Generation',
                    'Metric': 'Citation Accuracy',
                    'Value': f"{result.generation_metrics.citation_accuracy:.2%}"
                })

                # Performance
                metrics_data.append({
                    'Category': 'Performance',
                    'Metric': 'Avg Latency',
                    'Value': f"{result.e2e_metrics.average_latency:.2f}s"
                })

                df = pd.DataFrame(metrics_data)
                st.dataframe(df, use_container_width=True)

                # Download results
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(result.to_dict(), indent=2, default=str),
                    file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    elif eval_mode == "Ablation Study":
        st.header("🔬 Ablation Study")
        st.markdown("Compare different system configurations to understand component contributions")

        ablation_file = st.selectbox(
            "Select Ablation Config",
            [str(p) for p in Path("evals/datasets").glob("ablation*.json")]
        )

        test_file = st.selectbox(
            "Select Test Dataset",
            ["evals/datasets/example_test_set.json"] +
            [str(p) for p in Path("evals/datasets").glob("*.json") if "ablation" not in p.name]
        )

        if st.button("Run Ablation Study", type="primary"):
            with st.spinner("Running ablation study (this may take a while)..."):
                try:
                    test_data = load_test_dataset(test_file)

                    with open(ablation_file, 'r') as f:
                        ablation_configs = json.load(f)

                    results = evaluator.evaluate_ablation(test_data, ablation_configs)

                    st.success(f"✅ Evaluated {len(results)} configurations")

                    # Comparison chart
                    st.subheader("📊 Configuration Comparison")
                    fig = plot_ablation_comparison(results)
                    st.plotly_chart(fig, use_container_width=True)

                    # Results table
                    st.subheader("📋 Detailed Comparison")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    # Export
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison (CSV)",
                        data=csv,
                        file_name=f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Ablation study failed: {e}")
                    logger.exception("Ablation error")

    else:  # Dataset Generation
        st.header("🎲 Generate Synthetic Test Dataset")

        col1, col2 = st.columns(2)

        with col1:
            num_examples = st.number_input("Number of Examples", 10, 500, 100)
            difficulty_dist = st.multiselect(
                "Difficulty Levels",
                ["easy", "medium", "hard"],
                default=["easy", "medium", "hard"]
            )

        with col2:
            question_types = st.multiselect(
                "Question Types",
                ["factual", "how_to", "why", "comparison", "calculation"],
                default=["factual", "how_to", "why"]
            )

        pdf_folder = st.text_input("PDF Folder Path", value="./data")

        if st.button("Generate Dataset", type="primary"):
            with st.spinner("Generating synthetic dataset..."):
                try:
                    generator = load_dataset_generator()

                    # Generate from all PDFs in folder
                    pdf_files = list(Path(pdf_folder).glob("*.pdf"))

                    if not pdf_files:
                        st.error(f"No PDF files found in {pdf_folder}")
                        return

                    all_examples = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, pdf_path in enumerate(pdf_files):
                        status_text.text(f"Processing {pdf_path.name}...")

                        examples = generator.generate_from_pdf(
                            pdf_path,
                            num_examples=num_examples // len(pdf_files),
                            difficulty_distribution={d: 1.0/len(difficulty_dist) for d in difficulty_dist},
                            question_type_distribution={qt: 1.0/len(question_types) for qt in question_types}
                        )

                        all_examples.extend([ex.to_dict() for ex in examples])
                        progress_bar.progress((idx + 1) / len(pdf_files))

                    status_text.text("Generation complete!")

                    st.success(f"✅ Generated {len(all_examples)} examples")

                    # Preview
                    st.subheader("📋 Preview")
                    st.json(all_examples[:3])

                    # Save
                    if st.button("💾 Save Dataset"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = Path(f"evals/datasets/synthetic_{timestamp}.json")

                        with open(output_file, 'w') as f:
                            json.dump(all_examples, f, indent=2)

                        st.success(f"Dataset saved to {output_file}")

                except Exception as e:
                    st.error(f"Dataset generation failed: {e}")
                    logger.exception("Dataset generation error")

    # Footer
    st.markdown("---")
    st.markdown("*World-Class RAG Evaluation Dashboard* | Built with Streamlit")


if __name__ == "__main__":
    main()
