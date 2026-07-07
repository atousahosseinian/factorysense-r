from pathlib import Path
import sys

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer


st.set_page_config(
    page_title="FactorySense-R",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 FactorySense-R")
st.subheader("Industrial Anomaly Detection Data Explorer")

st.markdown(
    """
This dashboard helps us understand the dataset before running any anomaly detection model.

In industrial anomaly detection, the model usually learns from normal / good images and is evaluated on both normal and defective images.
"""
)

dataset_root = st.sidebar.text_input(
    "Dataset root",
    value="data/mvtec",
)

explorer = MVTecDatasetExplorer(dataset_root)
categories = explorer.categories()

if not categories:
    st.warning("No MVTec AD categories found yet.")

    st.markdown(
        """
Expected local dataset structure:

data/mvtec/
  bottle/
    train/
      good/
    test/
      good/
      broken_large/
      contamination/
    ground_truth/
      broken_large/
        image_mask.png

For now, this is fine. The project structure is ready.
Next, place one MVTec AD category inside data/mvtec/.
"""
    )
    st.stop()

category = st.sidebar.selectbox("Category", categories)

df = explorer.dataframe(category)
summary = explorer.summary(category)

st.markdown(f"## Category: `{category}`")

col1, col2, col3, col4 = st.columns(4)

total_images = len(df)
train_images = int((df["split"] == "train").sum()) if not df.empty else 0
test_images = int((df["split"] == "test").sum()) if not df.empty else 0
defective_images = int((df["label"] == 1).sum()) if not df.empty else 0

col1.metric("Total Images", total_images)
col2.metric("Train Images", train_images)
col3.metric("Test Images", test_images)
col4.metric("Defective Images", defective_images)

st.markdown("## Dataset Summary")
st.dataframe(summary, use_container_width=True)

st.markdown("## Image Gallery")

split_options = sorted(df["split"].unique()) if not df.empty else []
selected_split = st.selectbox("Split", split_options)

filtered = df[df["split"] == selected_split]

defect_options = sorted(filtered["defect_type"].unique()) if not filtered.empty else []
selected_defect_type = st.selectbox("Defect type", defect_options)

filtered = filtered[filtered["defect_type"] == selected_defect_type]

max_images = st.slider(
    "Number of images to display",
    min_value=1,
    max_value=24,
    value=8,
)

sample_df = filtered.head(max_images)

if sample_df.empty:
    st.info("No images available for this selection.")
    st.stop()

cols = st.columns(4)

for idx, row in enumerate(sample_df.to_dict("records")):
    image_path = row["path"]
    mask_path = row["mask_path"]

    with cols[idx % 4]:
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption=Path(image_path).name, use_container_width=True)

        st.caption(f"Status: {row['status']} | Defect: {row['defect_type']}")

        if mask_path:
            mask = Image.open(mask_path).convert("RGB")
            st.image(mask, caption="Ground Truth Mask", use_container_width=True)
