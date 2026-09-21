import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS (UNCHANGED)
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "plant_disease_best.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"


# ============================================================
# MODEL CONFIGURATION (UNCHANGED)
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 281
LAST_CONV_LAYER = "top_activation"


# ============================================================
# FINAL MODEL METRICS (FALLBACK) (UNCHANGED)
# ============================================================

FALLBACK_METRICS = {
    "Test Accuracy": 81.49,
    "Top-5 Accuracy": 94.20,
    "Macro Precision": 44.65,
    "Macro Recall": 42.74,
    "Macro F1": 42.28,
}


# ============================================================
# REFINED NATURAL SAGE CANVAS & CRISP BOTANICAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* Natural Sage-Linen Canvas to eliminate glare and wash-out */
    .stApp {
        background: linear-gradient(180deg, #eaf2eb 0%, #e1ede3 50%, #ebf3ec 100%) !important;
        color: #16361d;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Balanced Container Width */
    .block-container {
        max-width: 1450px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Sidebar - Rich Environmental Woodland Green */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3c23 0%, #122817 100%) !important;
        border-right: 1px solid #1d4527;
    }
    [data-testid="stSidebar"] * {
        color: #e2f1e4 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* Hero Banner - Lush Natural Gradient */
    .hero-box {
        background: linear-gradient(135deg, #245e34 0%, #2f7a44 55%, #3d9255 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(35, 92, 48, 0.14);
        color: #ffffff;
    }
    .hero-badge {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #1a4222;
        background-color: #d8eedb;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin-bottom: 0.7rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        line-height: 1.6;
        color: #e1f5e4;
        max-width: 900px;
        margin: 0;
    }

    /* Crisp White Cards that Pop Against the Canvas */
    .info-card {
        background-color: #ffffff !important;
        border: 1px solid #c5dfcb !important;
        border-left: 5px solid #2f7a44 !important;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.4rem;
        color: #214426;
        font-size: 0.92rem;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(22, 54, 28, 0.06) !important;
    }

    /* Analysis Status Bar */
    .result-badge {
        background-color: #ffffff !important;
        border: 1px solid #bddcb2 !important;
        border-radius: 10px;
        padding: 0.75rem 1.3rem;
        margin-bottom: 1.2rem;
        font-weight: 700;
        color: #174b21;
        font-size: 0.92rem;
        box-shadow: 0 3px 10px rgba(22, 54, 28, 0.04);
    }

    /* Diagnosis Metric Card */
    .pred-card {
        background-color: #ffffff !important;
        border: 1px solid #bddbbd !important;
        border-top: 5px solid #2f7a44 !important;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 5px 18px rgba(22, 54, 28, 0.06) !important;
    }
    .pred-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #55795c;
        font-weight: 800;
    }
    .pred-name {
        font-size: 1.65rem;
        font-weight: 850;
        color: #174020;
        margin: 0.3rem 0 0.7rem 0;
        line-height: 1.25;
    }
    .pred-conf {
        font-size: 2.2rem;
        font-weight: 850;
        color: #276f3a;
        margin: 0;
    }

    /* Diagnostic Context Card */
    .space-filler-card {
        background: #ffffff !important;
        border: 1px solid #c9decb !important;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 14px rgba(22, 54, 28, 0.05) !important;
    }
    .space-filler-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #1e4d26;
        margin-bottom: 0.5rem;
    }
    .space-filler-item {
        font-size: 0.85rem;
        color: #36553b;
        line-height: 1.5;
        margin-bottom: 0.35rem;
    }

    /* Constrain Leaf Image Height */
    [data-testid="stImage"] img {
        max-height: 420px !important;
        object-fit: contain !important;
        border-radius: 12px !important;
        background-color: #ffffff;
        border: 1px solid #cbe0ce;
    }

    /* High-Contrast Crisp Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #c5dfcb !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 14px rgba(22, 54, 28, 0.05) !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #174020 !important;
        font-size: 1.75rem !important;
        font-weight: 850 !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #4b6f52 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }

    /* Light Environmental File Uploader Container */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #9fd2a6 !important;
        border-radius: 14px !important;
        box-shadow: 0 3px 12px rgba(22, 54, 28, 0.04);
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #2f7a44 !important;
    }

    /* Technology Pills */
    .tech-tag {
        display: inline-block;
        background-color: #e5f2e7;
        border: 1px solid #b7d7bb;
        color: #1d4b24;
        border-radius: 30px;
        padding: 0.38rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 0.22rem;
    }

    /* Disclaimer Section */
    .disclaimer-box {
        background-color: #ffffff;
        border: 1px solid #fae8be;
        border-left: 5px solid #d89f36;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        color: #614d24;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 1.8rem;
        box-shadow: 0 4px 14px rgba(97, 77, 36, 0.05);
    }

    /* Button and Progress Bars */
    .stButton > button {
        background-color: #286b3a !important;
        border-color: #286b3a !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    .stButton > button:hover {
        background-color: #1f562e !important;
        border-color: #1f562e !important;
    }
    .stProgress > div > div > div > div {
        background-color: #2e7a41 !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        padding-top: 2rem;
        color: #5c7e63;
        font-size: 0.85rem;
        border-top: 1px solid #cadfcd;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS (UNCHANGED)
# ============================================================

def clean_class_name(name):
    return " ".join(str(name).strip().split())


def format_percent(value):
    try:
        value = float(value)
        if value <= 1.0:
            value *= 100.0
        return value
    except Exception:
        return None


def recursive_find_metric(data, possible_keys):
    if isinstance(data, dict):
        normalized = {
            str(k).lower().replace("_", " ").replace("-", " ").strip(): v
            for k, v in data.items()
        }
        for wanted in possible_keys:
            wanted_normalized = wanted.lower().replace("_", " ").replace("-", " ").strip()
            if wanted_normalized in normalized:
                return normalized[wanted_normalized]

        for value in data.values():
            result = recursive_find_metric(value, possible_keys)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = recursive_find_metric(item, possible_keys)
            if result is not None:
                return result

    return None


# ============================================================
# LOAD MODEL & ASSETS (UNCHANGED)
# ============================================================

@st.cache_resource(show_spinner="Loading trained EfficientNetB0 model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found:\n{CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = data.get("class_mapping", data)
    class_names = []
    for i in range(NUM_CLASSES):
        key = str(i)
        if key not in mapping:
            raise KeyError(f"Class ID {key} is missing from class_names.json")
        class_names.append(clean_class_name(mapping[key]))
    return class_names


@st.cache_data
def load_model_metrics():
    metrics = FALLBACK_METRICS.copy()
    if not METADATA_PATH.exists():
        return metrics

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metric_search = {
            "Test Accuracy": ["test accuracy", "test_accuracy", "accuracy"],
            "Top-5 Accuracy": ["top-5 accuracy", "top 5 accuracy", "top5 accuracy", "top_5_accuracy"],
            "Macro Precision": ["macro precision", "macro_precision"],
            "Macro Recall": ["macro recall", "macro_recall"],
            "Macro F1": ["macro f1", "macro_f1", "f1", "f1 score", "f1_score"],
        }

        for metric_name, possible_keys in metric_search.items():
            value = recursive_find_metric(metadata, possible_keys)
            if value is not None:
                formatted = format_percent(value)
                if formatted is not None:
                    metrics[metric_name] = formatted
    except Exception:
        pass

    return metrics


# ============================================================
# PREPROCESSING & GRAD-CAM (UNCHANGED)
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def generate_gradcam(model, img_array, class_index):
    base_model = model.get_layer("efficientnetb0")
    target_layer = base_model.get_layer(LAST_CONV_LAYER)

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
    )

    gap = model.get_layer("global_average_pooling")
    bn = model.get_layer("batch_normalization")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("predictions")

    with tf.GradientTape() as tape:
        conv_outputs, base_features = feature_model(img_array)
        x = gap(base_features)
        x = bn(x, training=False)
        x = dropout(x, training=False)
        predictions = classifier(x)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))[0]
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

    return heatmap.numpy()


def create_gradcam_figure(original_image, heatmap):
    original = np.asarray(original_image.convert("RGB"))
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        original_image.size, Image.Resampling.BILINEAR
    )
    heatmap_array = np.asarray(heatmap_image)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(original)
    ax.imshow(heatmap_array, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌿 Plant Disease AI")
    st.caption("Deep Learning Agricultural Diagnostic System")
    st.divider()

    st.markdown("**Model Specifications**")
    st.markdown("• **Architecture:** EfficientNetB0")
    st.markdown("• **Pretraining:** ImageNet")
    st.markdown("• **Input Resolution:** 224 × 224 × 3")
    st.markdown("• **Total Classes:** 281")
    st.markdown("• **Explainability:** Grad-CAM")

    st.divider()
    st.markdown("**Core Tech Stack**")
    st.markdown("• Python\n• TensorFlow / Keras\n• NumPy & Pandas\n• Scikit-learn\n• Matplotlib\n• Streamlit")

    st.divider()
    st.markdown("**Workflow Stages**")
    st.markdown("1. Dataset Preprocessing & Augmentation\n2. Transfer Learning (EfficientNetB0)\n3. Model Fine-Tuning & Evaluation\n4. Grad-CAM Interpretability\n5. Streamlit Cloud/Local Inference")

    st.caption("Deep Learning Environmental Portfolio Project")


# ============================================================
# INITIALIZATION CHECK
# ============================================================

try:
    model = load_model()
    class_names = load_class_names()
    metrics = load_model_metrics()
except Exception as e:
    st.error("The application could not load the required model files.")
    st.code(str(e))
    st.stop()


# ============================================================
# HERO & OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <span class="hero-badge">PLANT HEALTH • COMPUTER VISION • SUSTAINABLE AGRI-TECH</span>
        <h1 class="hero-title">🌿 Plant Disease Intelligence</h1>
        <p class="hero-subtitle">
            An intelligent ecological diagnostic system built on <b>EfficientNetB0</b> transfer learning.
            Classifies leaf diseases across <b>281 distinct conditions</b>, backed by Grad-CAM visual explanations
            for agronomic transparency.
        </p>
    </div>
    <div class="info-card">
        🌱 <b>How It Works:</b> Upload an image of an affected plant leaf. The vision network inspects lesion patterns,
        determines the primary botanical pathogen, ranks candidate conditions in a <b>Top-5 prediction matrix</b>,
        and provides an activation heatmap revealing the exact leaf regions guiding the diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Analyze a Plant Leaf")
st.caption("Upload a JPG, JPEG, or PNG photograph. For best diagnostic accuracy, keep the leaf in focus and clear of heavy background clutter.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Upload a plant leaf image to run classification.",
)


# ============================================================
# PREDICTION & RESULTS (TIGHT COMPACT LAYOUT)
# ============================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    with st.spinner("Analyzing botanical features with EfficientNetB0..."):
        predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:5]
    predicted_index = int(top_indices[0])
    predicted_class = clean_class_name(class_names[predicted_index])
    confidence = float(predictions[predicted_index])

    st.markdown(
        """
        <div class="result-badge">
            ✅ <span>Analysis Complete &nbsp;•&nbsp; 281-Class Softmax Output &nbsp;•&nbsp; EfficientNetB0 Backbone</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2-Column layout: Leaf preview alongside Diagnosis & Summary
    img_col, pred_col = st.columns([1, 1.15], gap="large")

    with img_col:
        with st.container(border=True):
            st.markdown("**🖼️ Uploaded Leaf Sample**")
            st.image(image, use_container_width=True)
            st.caption(f"Source: {uploaded_file.name} | Original Size: {image.size[0]}×{image.size[1]} px")

    with pred_col:
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-label">Diagnosed Condition</div>
                <div class="pred-name">{predicted_class}</div>
                <div class="pred-label">Model Confidence</div>
                <div class="pred-conf">{confidence * 100:.2f}%</div>
                <div style="font-size:0.82rem; color:#507858; margin-top:0.35rem;">
                    Normalized probability output from final classification layer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence >= 0.70:
            st.success("🟢 **High Confidence:** Pathological features match training representations strongly.")
        elif confidence >= 0.40:
            st.warning("🟡 **Moderate Confidence:** Mixed symptoms detected; verify with the Top-5 distribution below.")
        else:
            st.info("🔵 **Low Confidence:** Unfamiliar presentation. Ensure the leaf is well-lit and unobstructed.")

        st.markdown(
            """
            <div class="space-filler-card">
                <div class="space-filler-title">📋 Diagnostic Summary & Next Steps</div>
                <div class="space-filler-item">🔍 <b>Visual Heatmap:</b> Scroll down to inspect the Grad-CAM lesion focus.</div>
                <div class="space-filler-item">🌾 <b>Cross-Check:</b> Compare secondary candidates in the Top-5 probability matrix.</div>
                <div class="space-filler-item">🛡️ <b>Agronomic Advisory:</b> Confirm with local agricultural extension guidelines before applying chemical interventions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Top-5 Predictions
    st.subheader("🏆 Top-5 Ranked Conditions")
    st.caption("Alternative matching pathogen classes ranked by softmax posterior probability.")

    top_col, info_col = st.columns([1.3, 0.7], gap="large")

    with top_col:
        with st.container(border=True):
            for rank, idx in enumerate(top_indices, start=1):
                prob = float(predictions[idx])
                name = clean_class_name(class_names[idx])
                c1, c2, c3 = st.columns([0.1, 0.72, 0.18])
                c1.write(f"**#{rank}**")
                c2.write(f"{name}")
                c3.write(f"**{prob * 100:.2f}%**")
                st.progress(prob)

    with info_col:
        with st.container(border=True):
            st.markdown("**💡 Probability Interpretation**")
            st.markdown(
                """
                - **Top-1 Match:** Primary predicted disease label.
                - **Top-5 Alternatives:** Closely related conditions sharing visual leaf spot or necrosis markers.
                - **High Dispersion:** Suggests symptoms shared across related fungal or bacterial strains.
                """
            )

    st.divider()

    # Grad-CAM Visualization
    st.subheader("🧠 Explainable AI — Grad-CAM Heatmap")
    st.caption("Gradient-weighted Class Activation Mapping (Grad-CAM) illuminates the specific leaf regions that activated the model's decision.")

    try:
        with st.spinner("Computing class activation gradients..."):
            heatmap = generate_gradcam(model, img_array, predicted_index)
            gradcam_fig = create_gradcam_figure(image, heatmap)

        gcol1, gcol2 = st.columns(2, gap="large")
        with gcol1:
            with st.container(border=True):
                st.markdown("**Original Leaf Specimen**")
                st.image(image, use_container_width=True)
        with gcol2:
            with st.container(border=True):
                st.markdown("**Grad-CAM Region Heatmap**")
                st.pyplot(gradcam_fig, clear_figure=True, use_container_width=True)

    except Exception as e:
        st.warning("Could not generate Grad-CAM visualization for this image.")
        with st.expander("View trace"):
            st.code(str(e))

    st.divider()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.subheader("📊 Model Validation Metrics")
st.caption("Benchmark scores calculated on the held-out test evaluation split.")

metric_names = [
    "Test Accuracy", "Top-5 Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
]
metric_cols = st.columns(5)

for col, m_name in zip(metric_cols, metric_names):
    val = metrics.get(m_name, 0.0)
    with col:
        st.metric(label=m_name, value=f"{val:.2f}%")


# ============================================================
# HIGHLIGHTS & TECH STACK
# ============================================================

st.subheader("🛠️ Project Highlights")
h_data = [
    ("🧹", "Data Pipelines", "Image preprocessing, standardization to 224×224 px, and tf.data pipelines."),
    ("🧠", "Transfer Learning", "ImageNet-pretrained EfficientNetB0 backbone fine-tuned on crop pathology."),
    ("🎯", "Specialized Fine-Tuning", "Deep convolutional block unfreezing adapted across 281 condition classes."),
    ("📈", "Evaluation Rigor", "Comprehensive evaluation tracking Top-1, Top-5, and class-balanced macro F1."),
    ("🔎", "Model Interpretability", "Grad-CAM saliency heatmaps verify lesion focus over background noise."),
    ("🚀", "End-to-End Delivery", "Lightweight, responsive Streamlit dashboard configured for rapid inference."),
]

h_cols = st.columns(3)
for i, (icon, title, desc) in enumerate(h_data):
    with h_cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)

st.subheader("💻 Environmental AI Tech Stack")
tech_stack = [
    "Python", "TensorFlow", "Keras", "EfficientNetB0", "NumPy",
    "Pandas", "Scikit-learn", "Matplotlib", "Computer Vision",
    "Transfer Learning", "Grad-CAM", "Streamlit"
]
pills_html = "".join([f'<span class="tech-tag">🌿 {tech}</span>' for tech in tech_stack])
st.markdown(f'<div style="margin-bottom:1.5rem;">{pills_html}</div>', unsafe_allow_html=True)


# ============================================================
# ABOUT PROJECT & PIPELINE METHODOLOGY
# ============================================================

with st.expander("📘 View Project Overview & Pipeline Methodology"):
    st.markdown("""
    ### Project Objective
    End-to-end deep learning system for automated plant leaf disease diagnosis across **281 disease classes**.

    ### Dataset Scope
    Trained on a master dataset of approximately **66,700 images** across **42 plant species**.

    ### Processing Pipeline
    1. **Input Leaf Image:** Converted to 3-channel RGB.
    2. **Resize:** Standardized to (224, 224, 3) using bilinear interpolation.
    3. **EfficientNetB0 Backbone:** Deep convolutional feature extraction.
    4. **Softmax Head:** 281-dimensional normalized probability distribution.
    5. **Explainable AI:** Grad-CAM computed using gradients from `top_activation`.
    """)

with st.expander("🔬 View Technical Architecture"):
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("""
        **Neural Network Configuration:**
        - Backbone: EfficientNetB0
        - Pretrained Weights: ImageNet
        - Resolution: 224 × 224 × 3
        - Output Classes: 281
        """)
    with tcol2:
        st.markdown("""
        **Deployment Environment:**
        - Deep Learning: TensorFlow / Keras
        - Image Processing: PIL, NumPy
        - Visual Analytics: Matplotlib
        - Web Framework: Streamlit
        """)


# ============================================================
# DISCLAIMER & FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer-box">
        <b>⚠️ Agricultural Advisory Disclaimer:</b> This application serves as a machine learning and portfolio demonstration.
        Classifications reflect statistical correlations extracted by deep neural networks and must not replace professional
        agronomist consultation for commercial crop treatment or chemical application.
    </div>
    <div class="footer-text">
        🌿 <b>Plant Disease Intelligence</b> • Ecological Deep Learning Project • EfficientNetB0 • Grad-CAM • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)