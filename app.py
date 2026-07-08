from pathlib import Path
import sys
import pandas as pd

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from factorysense.data.mvtec_loader import MVTecDatasetExplorer
from factorysense.models.simple_baseline import SimpleDifferenceAnomalyDetector
from factorysense.models.model_loader import load_detector
from factorysense.reporting.csv_report import summarize_inspection_report
from factorysense.robustness.robustness_runner import (
    build_robustness_report,
    summarize_robustness_report,
)
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
- Single-image inspection
- Batch inspection with CSV-style reporting
- Robustness testing under rotation, brightness, and contrast shifts
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

tab_data, tab_inspector, tab_batch, tab_robustness, tab_model_comparison = st.tabs(
    [
        "Data Explorer",
        "Simple Baseline Inspector",
        "Batch Inspection",
        "Robustness Tests",
        "Model Comparison",
    ]
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

    inspector_split_options = sorted(inspector_df["split"].unique())
    inspector_split = st.selectbox(
        "Inspector split",
        inspector_split_options,
        index=1 if "test" in inspector_split_options else 0,
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


with tab_batch:
    st.markdown("## Batch Inspection")

    st.info(
        "Run inspection on all images in a selected split and generate a quality analytics table."
    )

    if not categories:
        st.warning("No dataset category found. Create or add an MVTec-style dataset first.")
        st.stop()

    if not Path(model_path).exists():
        st.error(
            f"Model file not found: {model_path}. Train it first with scripts/02_train_simple_baseline.py"
        )
        st.stop()

    batch_category = st.selectbox(
        "Batch category",
        categories,
        key="batch_category",
    )

    batch_df = explorer.dataframe(batch_category)

    if batch_df.empty:
        st.warning("No images found for this category.")
        st.stop()

    batch_split_options = sorted(batch_df["split"].unique())
    batch_split = st.selectbox(
        "Batch split",
        batch_split_options,
        index=1 if "test" in batch_split_options else 0,
        key="batch_split",
    )

    batch_filtered = batch_df[batch_df["split"] == batch_split].copy()

    st.markdown(f"Images selected for batch inspection: **{len(batch_filtered)}**")

    if st.button("Run Batch Inspection"):
        model = SimpleDifferenceAnomalyDetector.load(model_path)

        rows = []

        progress = st.progress(0)

        for idx, row in enumerate(batch_filtered.to_dict("records")):
            image_path = row["path"]

            result = model.predict(image_path)
            binary_mask = model.binary_mask(image_path)

            defect_area_percent = float(binary_mask.mean() * 100)

            predicted_label = 1 if result["decision"] == "Reject" else 0
            true_label = int(row["label"])
            correct = predicted_label == true_label

            rows.append(
                {
                    "image_path": image_path,
                    "category": row["category"],
                    "split": row["split"],
                    "defect_type": row["defect_type"],
                    "true_label": true_label,
                    "status": row["status"],
                    "anomaly_score": result["anomaly_score"],
                    "threshold": result["threshold"],
                    "decision": result["decision"],
                    "predicted_label": predicted_label,
                    "defect_area_percent": defect_area_percent,
                    "correct": correct,
                }
            )

            progress.progress((idx + 1) / len(batch_filtered))

        import pandas as pd

        report_df = pd.DataFrame(rows)
        summary = summarize_inspection_report(report_df)

        st.markdown("### Batch Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Images", summary["total_images"])
        col2.metric("Rejected", summary["rejected_images"])
        col3.metric("Passed", summary["passed_images"])
        col4.metric("Defect Rate", f"{summary['defect_rate_percent']:.2f}%")

        col5, col6, col7 = st.columns(3)

        col5.metric("Mean Score", f"{summary['mean_anomaly_score']:.4f}")
        col6.metric("Mean Defect Area", f"{summary['mean_defect_area_percent']:.2f}%")

        if summary["accuracy"] is not None:
            col7.metric("Accuracy", f"{summary['accuracy']:.2%}")
        else:
            col7.metric("Accuracy", "N/A")

        st.markdown("### Batch Report")
        st.dataframe(report_df, use_container_width=True)

        csv_bytes = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV Report",
            data=csv_bytes,
            file_name=f"{batch_category}_{batch_split}_inspection_report.csv",
            mime="text/csv",
        )

        st.markdown("### Error Analysis Preview")

        errors_df = report_df[report_df["correct"] == False]

        if errors_df.empty:
            st.success("No errors found in this batch.")
        else:
            st.warning(f"{len(errors_df)} errors found.")
            st.dataframe(errors_df, use_container_width=True)

with tab_robustness:
    st.markdown("## Robustness Tests")

    st.info(
        "Test how stable the anomaly detector is under real-world shifts such as rotation, brightness, and contrast changes."
    )

    if not categories:
        st.warning("No dataset category found. Create or add an MVTec-style dataset first.")
        st.stop()

    if not Path(model_path).exists():
        st.error(
            f"Model file not found: {model_path}. Train it first with scripts/02_train_simple_baseline.py"
        )
        st.stop()

    robustness_category = st.selectbox(
        "Robustness category",
        categories,
        key="robustness_category",
    )

    robustness_df = explorer.dataframe(robustness_category)

    if robustness_df.empty:
        st.warning("No images found for this category.")
        st.stop()

    robustness_split_options = sorted(robustness_df["split"].unique())
    robustness_split = st.selectbox(
        "Robustness split",
        robustness_split_options,
        index=1 if "test" in robustness_split_options else 0,
        key="robustness_split",
    )

    robustness_filtered = robustness_df[robustness_df["split"] == robustness_split].copy()

    normal_only = st.checkbox(
        "Test only normal / good images",
        value=True,
        help="Useful for measuring false positives under real-world shifts.",
    )

    if normal_only:
        robustness_filtered = robustness_filtered[robustness_filtered["label"] == 0].copy()

    st.markdown(f"Images selected for robustness testing: **{len(robustness_filtered)}**")

    st.markdown("### Corruption Settings")

    col_rot, col_bright, col_contrast = st.columns(3)

    with col_rot:
        enable_rotation = st.checkbox("Rotation", value=True)
        rotation_values = st.multiselect(
            "Rotation angles",
            options=[5, -5, 10, -10, 90, 180, 270],
            default=[5, -5, 10, -10, 90, 180, 270],
        )

    with col_bright:
        enable_brightness = st.checkbox("Brightness", value=True)
        brightness_values = st.multiselect(
            "Brightness factors",
            options=[0.7, 0.85, 1.15, 1.3],
            default=[0.7, 0.85, 1.15, 1.3],
        )

    with col_contrast:
        enable_contrast = st.checkbox("Contrast", value=True)
        contrast_values = st.multiselect(
            "Contrast factors",
            options=[0.7, 0.85, 1.15, 1.3],
            default=[0.7, 0.85, 1.15, 1.3],
        )

    corruptions = [("original", 0.0)]

    if enable_rotation:
        for value in rotation_values:
            corruptions.append(("rotation", float(value)))

    if enable_brightness:
        for value in brightness_values:
            corruptions.append(("brightness", float(value)))

    if enable_contrast:
        for value in contrast_values:
            corruptions.append(("contrast", float(value)))

    if st.button("Run Robustness Tests"):
        if robustness_filtered.empty:
            st.error("No images available for the selected robustness test.")
            st.stop()

        model = SimpleDifferenceAnomalyDetector.load(model_path)

        with st.spinner("Running robustness tests..."):
            report_df = build_robustness_report(
                model=model,
                image_records_df=robustness_filtered,
                corruptions=corruptions,
            )

            summary_df = summarize_robustness_report(report_df)

        st.markdown("### Robustness Summary")

        if summary_df.empty:
            st.warning("No robustness results generated.")
            st.stop()

        original_row = summary_df[
            summary_df["corruption_type"].astype(str) == "original"
        ]

        shifted_rows = summary_df[
            summary_df["corruption_type"].astype(str) != "original"
        ]

        baseline_accuracy = (
            float(original_row["accuracy"].iloc[0])
            if not original_row.empty
            else None
        )

        worst_accuracy = (
            float(shifted_rows["accuracy"].min())
            if not shifted_rows.empty
            else None
        )

        max_reject_rate = (
            float(shifted_rows["reject_rate_percent"].max())
            if not shifted_rows.empty
            else None
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Tested Images", int(robustness_filtered.shape[0]))
        col2.metric(
            "Baseline Accuracy",
            "N/A" if baseline_accuracy is None else f"{baseline_accuracy:.2%}",
        )
        col3.metric(
            "Worst Shift Accuracy",
            "N/A" if worst_accuracy is None else f"{worst_accuracy:.2%}",
        )
        col4.metric(
            "Max Reject Rate",
            "N/A" if max_reject_rate is None else f"{max_reject_rate:.2f}%",
        )

        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### Robustness Interpretation")

        if normal_only and max_reject_rate is not None and max_reject_rate >= 90:
            st.error(
                "The model is highly sensitive to real-world shifts. Normal images are often rejected after rotation, brightness, or contrast changes."
            )
        elif worst_accuracy is not None and worst_accuracy < 0.8:
            st.warning(
                "The model performance drops under some shifts. This indicates limited robustness."
            )
        else:
            st.success(
                "The model appears stable under the selected shifts."
            )

        st.markdown("### Detailed Robustness Report")
        st.dataframe(report_df, use_container_width=True)

        summary_csv = summary_df.to_csv(index=False).encode("utf-8")
        detail_csv = report_df.to_csv(index=False).encode("utf-8")

        col_download1, col_download2 = st.columns(2)

        with col_download1:
            st.download_button(
                label="Download Robustness Summary CSV",
                data=summary_csv,
                file_name=f"{robustness_category}_{robustness_split}_robustness_summary.csv",
                mime="text/csv",
            )

        with col_download2:
            st.download_button(
                label="Download Detailed Robustness CSV",
                data=detail_csv,
                file_name=f"{robustness_category}_{robustness_split}_robustness_details.csv",
                mime="text/csv",
            )

        st.markdown("### Failure Cases")

        failure_df = report_df[report_df["correct"] == False]

        if failure_df.empty:
            st.success("No robustness failures found.")
        else:
            st.warning(f"{len(failure_df)} failure cases found.")
            st.dataframe(failure_df, use_container_width=True)

with tab_model_comparison:
    st.markdown("## Model Comparison")

    st.info(
        "Compare the educational simple baseline, PatchCore-style feature baseline, and rotation-augmented PatchCore-style model on the same dataset split."
    )

    if not categories:
        st.warning("No dataset category found. Create or add an MVTec-style dataset first.")
        st.stop()

    comparison_category = st.selectbox(
        "Comparison category",
        categories,
        key="comparison_category",
    )

    comparison_df = explorer.dataframe(comparison_category)

    if comparison_df.empty:
        st.warning("No images found for this category.")
        st.stop()

    comparison_split_options = sorted(comparison_df["split"].unique())
    comparison_split = st.selectbox(
        "Comparison split",
        comparison_split_options,
        index=1 if "test" in comparison_split_options else 0,
        key="comparison_split",
    )

    comparison_filtered = comparison_df[
        comparison_df["split"] == comparison_split
    ].copy()

    st.markdown(f"Images selected: **{len(comparison_filtered)}**")

    st.markdown("### Model Paths")

    simple_path = st.text_input(
        "Simple baseline model",
        value="models/simple_baseline_bottle.npz",
        key="comparison_simple_path",
    )

    patchcore_path = st.text_input(
        "PatchCore-style model",
        value="models/patchcore_style_bottle.npz",
        key="comparison_patchcore_path",
    )

    patchcore_aug_path = st.text_input(
        "Rotation-augmented PatchCore-style model",
        value="models/patchcore_style_aug_bottle.npz",
        key="comparison_patchcore_aug_path",
    )

    comparison_device = st.selectbox(
        "Device",
        ["cpu", "auto"],
        index=0,
        key="comparison_device",
    )

    selected_models = st.multiselect(
        "Models to compare",
        options=["simple", "patchcore_style", "patchcore_style_aug"],
        default=["simple", "patchcore_style", "patchcore_style_aug"],
    )

    model_path_map = {
        "simple": simple_path,
        "patchcore_style": patchcore_path,
        "patchcore_style_aug": patchcore_aug_path,
    }

    def run_single_model_comparison(model_name, model, df):
        rows = []

        for row in df.to_dict("records"):
            image_path = row["path"]

            result = model.predict(image_path)
            binary_mask = model.binary_mask(image_path)

            predicted_label = 1 if result["decision"] == "Reject" else 0
            true_label = int(row["label"])
            correct = predicted_label == true_label

            rows.append(
                {
                    "model_name": model_name,
                    "image_path": image_path,
                    "category": row["category"],
                    "split": row["split"],
                    "defect_type": row["defect_type"],
                    "true_label": true_label,
                    "status": row["status"],
                    "anomaly_score": result["anomaly_score"],
                    "threshold": result["threshold"],
                    "decision": result["decision"],
                    "predicted_label": predicted_label,
                    "defect_area_percent": float(binary_mask.mean() * 100),
                    "correct": correct,
                }
            )

        return pd.DataFrame(rows)

    if st.button("Run Clean Model Comparison"):
        all_reports = []
        summary_rows = []

        for selected_model in selected_models:
            selected_path = Path(model_path_map[selected_model])

            if not selected_path.exists():
                st.warning(f"Skipping {selected_model}: model not found at {selected_path}")
                continue

            with st.spinner(f"Running {selected_model}..."):
                model = load_detector(
                    model_name=selected_model,
                    model_path=selected_path,
                    device=comparison_device,
                )

                report_df = run_single_model_comparison(
                    model_name=selected_model,
                    model=model,
                    df=comparison_filtered,
                )

            summary = summarize_inspection_report(report_df)
            summary["model_name"] = selected_model
            summary_rows.append(summary)
            all_reports.append(report_df)

        if not all_reports:
            st.error("No model reports were generated.")
            st.stop()

        clean_report_df = pd.concat(all_reports, ignore_index=True)
        clean_summary_df = pd.DataFrame(summary_rows)

        clean_summary_df = clean_summary_df[
            ["model_name"] + [col for col in clean_summary_df.columns if col != "model_name"]
        ]

        st.markdown("### Clean Dataset Comparison Summary")
        st.dataframe(clean_summary_df, use_container_width=True)

        st.markdown("### Clean Dataset Detailed Report")
        st.dataframe(clean_report_df, use_container_width=True)

        st.download_button(
            label="Download Clean Comparison Summary CSV",
            data=clean_summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{comparison_category}_{comparison_split}_model_comparison_summary.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download Clean Comparison Detailed CSV",
            data=clean_report_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{comparison_category}_{comparison_split}_model_comparison_details.csv",
            mime="text/csv",
        )

    st.divider()

    st.markdown("## Robustness Model Comparison")

    normal_only_comparison = st.checkbox(
        "Robustness comparison on normal / good images only",
        value=True,
        key="robustness_comparison_normal_only",
    )

    robustness_comparison_df = comparison_filtered.copy()

    if normal_only_comparison:
        robustness_comparison_df = robustness_comparison_df[
            robustness_comparison_df["label"] == 0
        ].copy()

    st.markdown(f"Robustness images selected: **{len(robustness_comparison_df)}**")

    col_rot_cmp, col_bright_cmp, col_contrast_cmp = st.columns(3)

    with col_rot_cmp:
        cmp_enable_rotation = st.checkbox("Compare rotation", value=True)
        cmp_rotation_values = st.multiselect(
            "Compare rotation angles",
            options=[5, -5, 10, -10, 90, 180, 270],
            default=[5, -5, 10, -10, 90, 180, 270],
            key="cmp_rotation_values",
        )

    with col_bright_cmp:
        cmp_enable_brightness = st.checkbox("Compare brightness", value=True)
        cmp_brightness_values = st.multiselect(
            "Compare brightness factors",
            options=[0.7, 0.85, 1.15, 1.3],
            default=[0.7, 0.85, 1.15, 1.3],
            key="cmp_brightness_values",
        )

    with col_contrast_cmp:
        cmp_enable_contrast = st.checkbox("Compare contrast", value=True)
        cmp_contrast_values = st.multiselect(
            "Compare contrast factors",
            options=[0.7, 0.85, 1.15, 1.3],
            default=[0.7, 0.85, 1.15, 1.3],
            key="cmp_contrast_values",
        )

    comparison_corruptions = [("original", 0.0)]

    if cmp_enable_rotation:
        for value in cmp_rotation_values:
            comparison_corruptions.append(("rotation", float(value)))

    if cmp_enable_brightness:
        for value in cmp_brightness_values:
            comparison_corruptions.append(("brightness", float(value)))

    if cmp_enable_contrast:
        for value in cmp_contrast_values:
            comparison_corruptions.append(("contrast", float(value)))

    if st.button("Run Robustness Model Comparison"):
        all_robustness_reports = []
        all_robustness_summaries = []

        if robustness_comparison_df.empty:
            st.error("No images available for robustness comparison.")
            st.stop()

        for selected_model in selected_models:
            selected_path = Path(model_path_map[selected_model])

            if not selected_path.exists():
                st.warning(f"Skipping {selected_model}: model not found at {selected_path}")
                continue

            with st.spinner(f"Running robustness for {selected_model}..."):
                model = load_detector(
                    model_name=selected_model,
                    model_path=selected_path,
                    device=comparison_device,
                )

                report_df = build_robustness_report(
                    model=model,
                    image_records_df=robustness_comparison_df,
                    corruptions=comparison_corruptions,
                )

                summary_df = summarize_robustness_report(report_df)

            report_df.insert(0, "model_name", selected_model)
            summary_df.insert(0, "model_name", selected_model)

            all_robustness_reports.append(report_df)
            all_robustness_summaries.append(summary_df)

        if not all_robustness_reports:
            st.error("No robustness reports were generated.")
            st.stop()

        robustness_report_df = pd.concat(all_robustness_reports, ignore_index=True)
        robustness_summary_df = pd.concat(all_robustness_summaries, ignore_index=True)

        st.markdown("### Robustness Comparison Summary")
        st.dataframe(robustness_summary_df, use_container_width=True)

        st.markdown("### Key Interpretation")

        simple_shift_failures = robustness_summary_df[
            (robustness_summary_df["model_name"] == "simple")
            & (robustness_summary_df["corruption_type"] != "original")
            & (robustness_summary_df["reject_rate_percent"] >= 90)
        ]

        aug_rotation_success = robustness_summary_df[
            (robustness_summary_df["model_name"] == "patchcore_style_aug")
            & (robustness_summary_df["corruption_type"] == "rotation")
            & (robustness_summary_df["accuracy"] >= 0.99)
        ]

        if not simple_shift_failures.empty and not aug_rotation_success.empty:
            st.success(
                "The comparison shows the full project story: the simple baseline fails under shifts, while the rotation-augmented PatchCore-style model remains robust."
            )
        else:
            st.info(
                "Review the summary table to compare how each model behaves under each shift."
            )

        st.markdown("### Robustness Detailed Report")
        st.dataframe(robustness_report_df, use_container_width=True)

        st.download_button(
            label="Download Robustness Comparison Summary CSV",
            data=robustness_summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{comparison_category}_{comparison_split}_robustness_model_comparison_summary.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download Robustness Comparison Detailed CSV",
            data=robustness_report_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{comparison_category}_{comparison_split}_robustness_model_comparison_details.csv",
            mime="text/csv",
        )

