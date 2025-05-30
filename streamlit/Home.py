import streamlit as st

st.set_page_config(page_title="Home", layout="wide")
st.title("Welcome to the Stereoscopic 6D Pose Estimation Project")
st.markdown("""
Welcome to the S6DPE Evaluation Suite

This is the **home page** for navigating your 6D pose estimation project workspace.

From here, you can access comprehensive tools to **explore your models, datasets, and training performance**.

---

### Use the top navigation bar to explore:

- **Dashboard**  
  Compare training and validation metrics like loss, translation RMSE, and rotation RMSE across multiple training runs. Visualize model behavior over time with customizable plots.

- **Documentation**  
  Browse detailed evaluation reports (`.md` files) for each model version. These include training configuration, performance metrics, accuracy evaluations, and observations.

- **Implementation**  
  Go ahead and test out the models and see how they perform.

---

This is a personal project focused on **6D pose estimation** using stereo vision and deep learning — built from scratch, iterated through trial and error, and continuously improved through hands-on experimentation.

The goal: accurately predict an object’s **6 degrees of freedom (position + orientation)** from stereo image inputs using different ConvNeXt-based model variants.

You’ll find three core model types here:
- **Vanilla ConvNeXt** – basic single-view pipeline, the baseline
- **Shared-weight stereo models** – two ConvNeXt branches processing left/right images in parallel with shared weights
- **6-channel early fusion** – merging stereo views into a single input with 6 channels, fed into a single ConvNeXt

This dashboard is the central hub I built to:
- Monitor training loss and RMSE over time
- Compare different architectures side by side
- Store and browse evaluation results
- Understand how dataset size, batch size, and head architecture affect performance
- Document and keep track of every experiment

Everything’s logged, visualized, and exported (Messily).  

"""
)