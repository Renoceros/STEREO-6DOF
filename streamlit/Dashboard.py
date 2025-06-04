import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import torch

# --- Import Model Architectures ---
from utility.models.vanila_model import ConvNeXt6DP, PoseDataset
from utility.models.six_ch_model import ConvNeXt6DP6ch, Stereo6ChPoseDataset
from utility.models.sw_twin_model import StereoConvNeXt6DP, StereoPoseDataset
from utility.train_utils import make_head 
# ==== CONFIGURATION ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EVAL_PATH = os.path.join(MODEL_DIR, "EVAL.xlsx")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_CSV_PATH = os.path.join(DATASET_DIR, "dataset_info.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224 # Default image size for dummy input, should match trained model
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

def plot_correlation_heatmap(data, title="Correlation Heatmap"):
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def plot_scatter(df, x, y, hue=None):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="tab10", ax=ax)
    ax.set_title(f"{y} vs {x}" + (f" (hue={hue})" if hue else ""))
    st.pyplot(fig)


def plot_grouped_line(df, x, y, groupby):
    grouped = df.groupby([groupby, x])[y].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=grouped, x=x, y=y, hue=groupby, marker="o", ax=ax)
    ax.set_title(f"{y} vs {x} grouped by {groupby}")
    st.pyplot(fig)

def get_model_from_type_and_head(model_type_identifier, head_layers):
    """
    Instantiates the correct model architecture with specified head layers
    based on a string identifier derived from the model filename.
    """
    if "convnext6dp6ch" in model_type_identifier.lower():
        # 6-Channel Stacked model
        return ConvNeXt6DP6ch(head_layers=head_layers).to(DEVICE)
    elif "stereoconvnext6dp" in model_type_identifier.lower() or "sw" in model_type_identifier.lower():
        # Stereo Shared Weights (Twin Heads) model
        return StereoConvNeXt6DP(head_layers=head_layers).to(DEVICE)
    elif "convnext6dp" in model_type_identifier.lower():
        # Vanilla model (assuming it's the default if other identifiers aren't found)
        return ConvNeXt6DP(head_layers=head_layers).to(DEVICE)
    else:
        st.error(f"Could not determine model type from identifier: {model_type_identifier}")
        return None

def get_dummy_input(model_instance, img_size):
    """
    Generates a dummy input tensor(s) based on the model's expected input.
    """
    # Check the class name of the instantiated model to determine input type
    model_class_name = model_instance.__class__.__name__

    if model_class_name == "ConvNeXt6DP6ch":
        return torch.randn(1, 6, img_size, img_size).to(DEVICE)
    elif model_class_name == "StereoConvNeXt6DP":
        left = torch.randn(1, 3, img_size, img_size).to(DEVICE)
        right = torch.randn(1, 3, img_size, img_size).to(DEVICE)
        return (left, right) # Return tuple for multi-input models
    elif model_class_name == "ConvNeXt6DP":
        return torch.randn(1, 3, img_size, img_size).to(DEVICE)
    else:
        raise ValueError(f"Unknown model class for dummy input: {model_class_name}")

def list_clean_models(model_dir):
    """
    Lists all model files in the specified directory that start with 'CLEAN-' and end with '.pth'.
    """
    if not os.path.exists(model_dir):
        return []
    
    clean_models = [f for f in os.listdir(model_dir) if f.startswith("CLEAN-") and f.endswith(".pth")]
    return sorted(clean_models)

# ==== STREAMLIT UI ====
st.title("Dashboard : 6D Pose Model and Dataset Evaluation ")

# Tabs for layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([" Compare Runs", " Per-Model Reports", " Model Eval Sheet"," Dataset parameters Sheet ","Freeze Model" ])

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
        st.markdown("---")
        st.subheader("🔧 Controls")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_var = st.selectbox("X-axis variable", df.columns, key="x_var")
        with col2:
            y_var = st.selectbox("Y-axis variable", df.columns, key="y_var")
        with col3:
            hue_var = st.selectbox("Color by (hue)", [None] + list(df.columns), key="hue_var")
        with col4:
            group_line = st.selectbox("Group line plots by", df.columns, key="group_line")

        col5, col6 = st.columns(2)
        with col5:
            variant_filter = st.multiselect("Filter by VARIANT", sorted(df["VARIANT"].unique()), default=df["VARIANT"].unique())
        with col6:
            arch_filter = st.multiselect("Filter by HEAD_ARCH", sorted(df["HEAD_ARCH"].unique()), default=df["HEAD_ARCH"].unique())

        # Filter data
        filtered_df = df[
            df["VARIANT"].isin(variant_filter) &
            df["HEAD_ARCH"].isin(arch_filter)
        ]

        # ----------------- Visualizations
        st.subheader("📊 Correlation Heatmap")
        plot_correlation_heatmap(filtered_df)

        st.subheader("📉 Scatter Plot")
        plot_scatter(filtered_df, x_var, y_var, hue=hue_var)

        st.subheader("📈 Line Plot (Grouped)")
        try:
            if x_var == "ID" or y_var == "ID" or group_line == "ID":
                raise ValueError("ID is not a valid choice for grouped line plots.")
            plot_grouped_line(filtered_df, x=x_var, y=y_var, groupby=group_line)
        except Exception as e:
            st.error("⚠️ Cannot generate line plot. Please change X/Y/Group to something other than 'ID'.")

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

# ==== TAB 5 : Freeze Models ====
with tab5:
    st.header("Freeze Models")
    st.markdown("""
    Select a pre-trained `CLEAN` model and manually specify its head architecture to convert it into a TorchScript (`.pt`) file.
    TorchScript models are optimized for deployment and inference.
    """)

    # List available clean models
    available_models = list_clean_models(MODEL_DIR)

    if not available_models:
        st.warning(f"No `CLEAN-*.pth` models found in `{MODEL_DIR}`. Please train a model first or check the path.")
    else:
        col_freeze1, col_freeze2 = st.columns(2)

        with col_freeze1:
            selected_model_file = st.selectbox(
                "Select Model to Freeze",
                available_models,
                help="Choose a .pth model file starting with 'CLEAN-' from your model directory."
            )
            # You might want to parse IMG_SIZE from filename here if it's encoded,
            # otherwise rely on a general default or user input
            freeze_img_size = st.number_input("Image Size (pixels) for Freezing", min_value=64, value=IMG_SIZE, step=32, help="Must match the image size used during training of the selected model.")


        with col_freeze2:
            head_layers_str_freeze = st.text_input(
                "Manually Enter Head Layers (comma-separated integers)",
                value="512", # Default, but user MUST override to match actual model
                help="E.g., '512' for a single layer, '256,128' for multiple. THIS MUST EXACTLY MATCH THE HEAD ARCHITECTURE USED DURING TRAINING."
            )
            output_subdir = st.text_input(
                "Output Subdirectory for Baked Model",
                "baked",
                help="The TorchScript model will be saved to `model/<output_subdir>/`."
            )
            os.makedirs(os.path.join(MODEL_DIR, output_subdir), exist_ok=True) # Ensure output directory exists

        st.markdown("---")
        if st.button("🚀 Start Freezing Model"):
            if not selected_model_file:
                st.error("Please select a model file.")
            elif not head_layers_str_freeze.strip():
                st.error("Please enter the head layers for the selected model.")
            else:
                try:
                    head_layers_freeze = [int(x.strip()) for x in head_layers_str_freeze.split(',') if x.strip()]
                    if not head_layers_freeze:
                        st.error("Head Layers cannot be empty after parsing. Enter valid integers.")
                        st.stop()

                    status_placeholder = st.empty()
                    status_placeholder.info(f"Preparing to freeze model: {selected_model_file}")

                    # Determine model type from filename
                    # This logic assumes your filenames roughly contain model type identifiers
                    model_type_identifier = selected_model_file.lower()
                    
                    # Instantiate the model with the manually provided head architecture
                    model = get_model_from_type_and_head(model_type_identifier, head_layers_freeze)
                    if model is None:
                        st.error("Failed to instantiate model. Check model type identifier.")
                        st.stop()

                    # Load the state dictionary
                    model_path = os.path.join(MODEL_DIR, selected_model_file)
                    status_placeholder.info(f"Loading state dictionary from: {model_path}")
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    model.to(DEVICE)
                    model.eval() # Set model to evaluation mode

                    # Get dummy input for tracing
                    status_placeholder.info("Generating dummy input for TorchScript tracing...")
                    dummy_input = get_dummy_input(model, freeze_img_size)

                    # Perform TorchScript tracing
                    status_placeholder.info("Tracing model with TorchScript...")
                    scripted_model = torch.jit.trace(model, dummy_input)
                    
                    # Define output path
                    output_filename = selected_model_file.replace(".pth", ".pt")
                    baked_output_path = os.path.join(MODEL_DIR, output_subdir, output_filename)

                    # Save the TorchScript model
                    scripted_model.save(baked_output_path)
                    
                    status_placeholder.success(f"✅ Model successfully frozen and saved to: {baked_output_path}")
                    st.balloons()

                except Exception as e:
                    st.error(f"An error occurred during freezing: {e}")
                    st.exception(e) # Display full traceback for debugging
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        st.info("CUDA cache cleared.")