from pathlib import Path
import sys

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.visualization.heatmap import (
    anomaly_map_to_pil,
    binary_mask_to_pil,
    overlay_to_pil,
)


st.set_page_config(
    page_title="FactorySense-R",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 FactorySense-R")
st.subheader("Robust Industrial Anomaly Detection Under Real-World Shifts")

st.markdown(
    """
FactorySense-R is an educational and practical dashboard for industrial anomaly detection.

The current version includes:
- MVTec-style dataset exploration
- A simple educational anomaly detector
- Anomaly score, heatmap, binary mask, and Pass/Reject decision
"""
)

dataset_root = st.sidebar.text_input(
    "Dataset root",
    value="data/mvtec",
)

model_path = st.sidebar.text_input(
    "Simple baseline model path",
    value="models/simple_baseline_bottle.npz",
)

explorer = MVTecDatasetExplorer(dataset_root)
categories = explorer.categories()

tab_data, tab_inspector = st.tabs(
    ["Data Explorer", "Simple Baseline Inspector"]
)


with tab_data:
    st.markdown("## Data Explorer")

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

    st.markdown(f"### Category: `{category}`")

    col1, col2, col3, col4 = st.columns(4)

    total_images = len(df)
    train_images = int((df["split"] == "train").sum()) if not df.empty else 0
    test_images = int((df["split"] == "test").sum()) if not df.empty else 0
    defective_images = int((df["label"] == 1).sum()) if not df.empty else 0

    col1.metric("Total Images", total_images)
    col2.metric("Train Images", train_images)
    col3.metric("Test Images", test_images)
    col4.metric("Defective Images", defective_images)

    st.markdown("### Dataset Summary")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Image Gallery")

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
    else:
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


with tab_inspector:
    st.markdown("## Simple Baseline Inspector")

    st.info(
        "This is an educational baseline. It compares an input image with the average normal reference image."
    )

    if not categories:
        st.warning("No dataset category found. Create or add an MVTec-style dataset first.")
        st.stop()

    if not Path(model_path).exists():
        st.error(
            f"Model file not found: {model_path}. Train it first with scripts/02_train_simple_baseline.py"
        )
        st.stop()

    selected_category = st.selectbox(
        "Inspector category",
        categories,
        key="inspector_category",
    )

    inspector_df = explorer.dataframe(selected_category)

    if inspector_df.empty:
        st.warning("No images found for this category.")
        st.stop()

    inspector_split = st.selectbox(
        "Inspector split",
        sorted(inspector_df["split"].unique()),
        index=1 if "test" in sorted(inspector_df["split"].unique()) else 0,
        key="inspector_split",
    )

    inspector_filtered = inspector_df[inspector_df["split"] == inspector_split]

    inspector_defect_type = st.selectbox(
        "Inspector defect type",
        sorted(inspector_filtered["defect_type"].unique()),
        key="inspector_defect_type",
    )

    inspector_filtered = inspector_filtered[
        inspector_filtered["defect_type"] == inspector_defect_type
    ]

    image_options = inspector_filtered["path"].tolist()

    selected_image_path = st.selectbox(
        "Image",
        image_options,
        key="inspector_image",
    )

    if st.button("Run Inspection"):
        model = SimpleDifferenceAnomalyDetector.load(model_path)

        result = model.predict(selected_image_path)
        anomaly_map = model.anomaly_map(selected_image_path)
        binary_mask = model.binary_mask(selected_image_path)
        defect_area = float(binary_mask.mean() * 100)

        st.markdown("### Inspection Result")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
        metric_col2.metric("Threshold", f"{result['threshold']:.4f}")
        metric_col3.metric("Decision", result["decision"])
        metric_col4.metric("Defect Area", f"{defect_area:.2f}%")

        image_col1, image_col2, image_col3, image_col4 = st.columns(4)

        with image_col1:
            original = Image.open(selected_image_path).convert("RGB")
            st.image(original, caption="Original Image", use_container_width=True)

        with image_col2:
            heatmap = anomaly_map_to_pil(anomaly_map)
            st.image(heatmap, caption="Anomaly Heatmap", use_container_width=True)

        with image_col3:
            overlay = overlay_to_pil(selected_image_path, anomaly_map)
            st.image(overlay, caption="Overlay", use_container_width=True)

        with image_col4:
            mask = binary_mask_to_pil(binary_mask)
            st.image(mask, caption="Binary Mask", use_container_width=True)

        st.markdown("### Interpretation")

        if result["decision"] == "Reject":
            st.error(
                "The image was classified as Reject because its anomaly score is above the calibrated threshold."
            )
        else:
            st.success(
                "The image was classified as Pass because its anomaly score is below the calibrated threshold."
            )
