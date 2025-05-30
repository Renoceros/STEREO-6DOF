import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==== CONFIGURATION ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EVAL_PATH = os.path.join(MODEL_DIR, "EVAL.xlsx")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_CSV_PATH = os.path.join(DATASET_DIR, "dataset_info.csv")

st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

# ==== HELPER FUNCTIONS ====
def get_all_runs():
    return sorted([d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))])

def load_scalars_from_run(run_dir, tags):
    scalars = {}
    try:
        event = EventAccumulator(os.path.join(RUNS_DIR, run_dir))
        event.Reload()
        for tag in tags:
            if tag in event.Tags()['scalars']:
                steps, values = zip(*[(e.step, e.value) for e in event.Scalars(tag)])
                scalars[tag] = (steps, values)
    except Exception as e:
        print(f"⚠️ Skipping {run_dir} due to error: {e}")
    return scalars


def plot_selected_metric(metric, runs):
    fig, ax = plt.subplots(figsize=(10, 5))
    color_cycle = plt.cm.tab20.colors  # or tab10, Set1, etc.

    for i, run in enumerate(runs):
        scalars = load_scalars_from_run(run, [metric])
        if metric in scalars:
            steps, values = scalars[metric]
            ax.plot(steps, values, label=run, color=color_cycle[i % len(color_cycle)], linewidth=2)

    ax.set_title(f"{metric} across runs", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig

def load_model_markdown(md_file):
    with open(os.path.join(MODEL_DIR, md_file), "r") as f:
        return f.read()

def load_metadata_md(batch_id):
    md_path = os.path.join(DATASET_DIR, f"batch{batch_id}", f"metadata_{batch_id}.md")
    if os.path.exists(md_path):
        with open(md_path, 'r') as file:
            return file.read()
    return None


# ==== STREAMLIT UI ====
st.title("Dashboard : 6D Pose Model and Dataset Evaluation ")

# Tabs for layout
tab1, tab2, tab3, tab4 = st.tabs([" Compare Runs", " Per-Model Reports", " Model Eval Sheet"," Dataset parameters Sheet " ])

# ==== TAB 1: Compare Runs ====
with tab1:
    st.header(" Training and Validation Metrics")
    metric = st.selectbox("Select Metric", ["Loss/Train", "Loss/Val", "RMSE/Train_Translation", "RMSE/Train_Rotation", "RMSE/Val_Translation", "RMSE/Val_Rotation"])
    runs = get_all_runs()
    selected_runs = st.multiselect("Select Runs to Compare", runs, default=runs)

    if selected_runs:
        fig = plot_selected_metric(metric, selected_runs)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one run to display.")

# ==== TAB 2: Per-Model Reports ====
with tab2:
    st.header(" Evaluation Reports (.md)")
    md_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".md")])
    selected_md = st.selectbox("Choose a model report", md_files)

    if selected_md:
        md_content = load_model_markdown(selected_md)
        st.markdown(md_content)

# ==== TAB 3: Eval Sheet ====
with tab3:
    st.header(" Evaluation Summary Table (EVAL.xlsx)")
    if os.path.exists(EVAL_PATH):
        df = pd.read_excel(EVAL_PATH)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download as CSV", df.to_csv(index=False), "EVAL.csv")
        st.markdown(""" 
        In the Table there are 3 Variants of models:

        - **VANILA**: A standard single-image ConvNeXt pipeline trained on the left stereo image only. This serves as a baseline to compare stereo-based improvements. Input is a regular 3-channel (RGB) image, and the network predicts 6D pose directly.

        - **Double**: A shared-weight twin-branch stereo model. It splits the stereo image into left and right halves, runs each through the same ConvNeXt backbone (with shared weights), extracts features from both, and then concatenates the two feature vectors before feeding them into a pose regression head. This lets the model learn from stereo disparity without doubling the number of parameters.

        - **6CH**: An early fusion model that combines the left and right stereo views into a single 6-channel image (left RGB stacked with right RGB). This 6-channel image is passed through a modified ConvNeXt model with its first convolutional layer adjusted to accept 6 channels. This approach allows stereo information to be fused at the input level, encouraging the model to learn stereo relations directly in the feature maps.

        Each variant has been tested with different input resolutions (224 and 384), head architectures (e.g., 640→512→9 or 640→256→128→9), and batch sizes to explore the trade-offs in accuracy, VRAM usage, and training speed.
        Some variants are dropped after a couple of runs because of very poor performace in n-epoch compared to the others.
        """)
    else:
        st.error("EVAL.xlsx not found in model/ directory")

# ==== TAB 4: Dataset Parameters Sheet ====
with tab4:
    st.header("Dataset Parameters Sheet (dataset_info.csv)")
    
    # Dataset Info CSV
    if os.path.exists(DATASET_CSV_PATH):
        df = pd.read_csv(DATASET_CSV_PATH)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download as CSV", df.to_csv(index=False), "dataset_info.csv")
        # with st.expander("Unique Values per Column"):
        #     for col in df.columns:
        #         unique_vals = df[col].nunique()
        #         st.write(f"{col}: {unique_vals} unique values")
    else:
        st.error("❌ dataset_info.csv not found in model/ directory")

    # All Metadata Files
    st.subheader(" Dataset Metadata (All Batches)")
    for batch_id in sorted(os.listdir(os.path.join(DATASET_DIR))):
        if batch_id.startswith("batch"):
            batch_num = batch_id.replace("batch", "")
            md_path = os.path.join(DATASET_DIR, batch_id, f"metadata_{batch_num}.md")
            if os.path.exists(md_path):
                st.markdown(f"---\n### Metadata for dataset batch {batch_num}", unsafe_allow_html=True)
                with open(md_path, 'r') as file:
                    st.markdown(file.read(), unsafe_allow_html=True)

