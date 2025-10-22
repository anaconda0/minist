import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from pathlib import Path
from typing import Optional, Tuple, List

# ---- try to import canvas widget; fall back to upload if missing ----
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ModuleNotFoundError:
    HAS_CANVAS = False

# ================= Page config =================
st.set_page_config(page_title="MNIST — 3 Models Playground", page_icon="🧠", layout="wide")
st.title("🧠 MNIST: DNN vs CNN vs Decision Tree — Draw & Test")

# ================= Paths & sidebar =================
APP_DIR = Path(__file__).parent
st.sidebar.header("Models")
model_dir_str = st.sidebar.text_input(
    "Model directory",
    value=str((APP_DIR / "models").resolve()),
    help="Folder that contains the saved models."
)
MODELS_DIR = Path(model_dir_str).expanduser().resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_DT = "dt_mnist.joblib"

def find_dt_file(base: Path) -> Optional[Path]:
    pref = base / PREFERRED_DT
    if pref.exists():
        return pref
    matches = list(base.glob("dt*.joblib"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)

# --------- cache clear + reload button ---------
def clear_model_caches():
    try:
        load_keras_safe.clear()   # type: ignore[attr-defined]
        load_dt_safe.clear()      # type: ignore[attr-defined]
    except Exception:
        pass

if st.sidebar.button("🔄 Reload models"):
    clear_model_caches()
    st.experimental_rerun()

# ================= Safe loaders =================
@st.cache_resource(show_spinner=False)
def load_keras_safe(path: Path) -> Tuple[Optional[tf.keras.Model], str]:
    """Try tf.keras then standalone keras>=3. Returns (model, backend_tag or error tag)."""
    if not path or not path.exists():
        return None, "missing"

    tf_err = k3_err = None

    # 1) Try TensorFlow-bundled Keras
    try:
        mdl = tf_load_model(str(path), compile=False)
        return mdl, "tf.keras"
    except Exception as e:
        tf_err = e

    # 2) Try standalone Keras 3
    try:
        import keras as k3
        ver = getattr(k3, "__version__", "0")
        major = int(str(ver).split(".", 1)[0]) if ver else 0
        if major < 3:
            raise RuntimeError(
                f"Found keras=={ver}, but this file may need Keras 3+. "
                f'Install: pip install --upgrade "keras>=3"'
            )
        mdl = k3.models.load_model(str(path), compile=False)
        return mdl, "keras3"
    except Exception as e:
        k3_err = e

    st.sidebar.error(
        "Couldn't load Keras model.\n\n"
        f"- tf.keras: {tf_err}\n"
        f"- keras (standalone): {k3_err}\n\n"
        "If saved with Keras 3 (.keras), ensure:\n"
        '    pip install --upgrade "tensorflow==2.17.0" "keras==3.5.0" h5py'
    )
    return None, "error"

@st.cache_resource(show_spinner=False)
def load_dt_safe(path: Optional[Path]):
    if not path or not path.exists():
        return None
    try:
        return joblib.load(str(path))
    except Exception as e:
        st.sidebar.warning(f"Couldn't load DecisionTree model at {path.name}: {e}")
        return None

# ================= Resolve files =================
# DNN (single)
dnn_file = MODELS_DIR / "dnn_mnist.keras"
dnn, dnn_backend = (None, None)
if dnn_file.exists():
    dnn, dnn_backend = load_keras_safe(dnn_file)

# CNN (single or ensemble)
cnn_single_file = MODELS_DIR / "cnn_mnist.keras"
cnn_models: List[tf.keras.Model] = []
cnn_backends: List[str] = []
cnn_found_files: List[Path] = []     # what exists on disk
cnn_load_errors: List[Tuple[str, str]] = []  # (filename, backend_tag) that failed

if cnn_single_file.exists():
    cnn_found_files.append(cnn_single_file)
    mdl, be = load_keras_safe(cnn_single_file)
    if mdl is not None:
        cnn_models = [mdl]
        cnn_backends = [be]
    else:
        cnn_load_errors.append((cnn_single_file.name, be))
else:
    members = sorted(MODELS_DIR.glob("cnn_mnist-*.keras"), key=lambda p: p.name)
    cnn_found_files.extend(members)
    for p in members:
        mdl, be = load_keras_safe(p)
        if mdl is not None:
            cnn_models.append(mdl)
            cnn_backends.append(be)
        else:
            cnn_load_errors.append((p.name, be))

# Decision Tree
dt_path = find_dt_file(MODELS_DIR)
dt = load_dt_safe(dt_path)

# ================= Sidebar status =================
st.sidebar.subheader("Loaded Files")
st.sidebar.write(f"📁 Models dir: `{MODELS_DIR}`")

# Env check (handy)
with st.sidebar.expander("Environment check"):
    st.write("TensorFlow:", tf.__version__)
    try:
        import keras as _k3
        st.write("Keras:", _k3.__version__)
    except Exception:
        st.write("Keras: not installed as standalone (using tf.keras only)")

# DNN status
st.sidebar.write(
    "🧠 DNN: " +
    (f"`{dnn_file.name}` via **{dnn_backend}**" if dnn is not None else
     ("present but **failed to load**" if dnn_file.exists() else "— not found"))
)

# CNN status (show presence vs load)
if len(cnn_found_files) == 0:
    st.sidebar.write("🧠 CNN: — not found")
else:
    if len(cnn_models) == 0:
        names = ", ".join(f"`{p.name}`" for p in cnn_found_files)
        st.sidebar.write(f"🧠 CNN: present but **failed to load** → {names}")
        if cnn_load_errors:
            bad_name, backend = cnn_load_errors[0]
            st.sidebar.warning(f"Could not load `{bad_name}` (backend tried: {backend}).")
        st.sidebar.info(
            "Tip: Keras 3 .keras files usually require:\n"
            '• tensorflow==2.17.0\n'
            '• keras==3.5.0\n'
            "Then click **Reload models**."
        )
    elif len(cnn_models) == 1:
        name = cnn_found_files[0].name
        st.sidebar.write(f"🧠 CNN: `{name}` via **{cnn_backends[0]}**")
    else:
        st.sidebar.write(f"🧠 CNN: {len(cnn_models)} loaded (ensemble)")
        st.sidebar.write(" " + ", ".join(f"`{p.name}`" for p in cnn_found_files))

# DT status
st.sidebar.write("🌳 DecisionTree: " + (f"`{dt_path.name}`" if dt_path and dt is not None else
                                        ("present but **failed to load**" if dt_path else "— not found")))

# Determine which predictors are available
available = []
if dnn is not None: available.append("DNN")
if len(cnn_models) > 0: available.append("CNN")
if dt is not None: available.append("DecisionTree")

if not available:
    st.error(
        "No models loaded.\n\n"
        "• Put `dnn_mnist.keras` and/or `cnn_mnist.keras` (or `cnn_mnist-*.keras`) in the models folder.\n"
        "• Put a scikit-learn file named `dt_mnist.joblib` (or any `dt*.joblib`)."
    )
    st.stop()

mode = st.sidebar.radio("Choose prediction mode", ["Single model", "Compare all"])
chosen = st.sidebar.selectbox("Model", options=available) if mode == "Single model" else "Compare all"

st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width", 8, 40, 22)
bg_color = st.sidebar.color_picker("Background color", "#000000")
stroke_color = st.sidebar.color_picker("Pen color", "#FFFFFF")
realtime = st.sidebar.checkbox("Realtime predict", value=True)

# ================= Input (canvas or upload) =================
st.subheader("✍️ Draw a digit (0–9)")
input_img = None

if HAS_CANVAS:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=realtime,
    )
    if st.button("🧹 Clear"):
        st.rerun()
    if canvas_result is not None and canvas_result.image_data is not None:
        input_img = canvas_result.image_data
else:
    st.warning(
        "The drawing widget `streamlit-drawable-canvas` is not installed.\n\n"
        "• Add `streamlit-drawable-canvas==0.9.3` to requirements.txt (recommended), or\n"
        "• Upload a 28×28 or larger digit image below."
    )
    uploaded = st.file_uploader("Upload a PNG/JPG of a single digit", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            pil = Image.open(uploaded).convert("RGBA")
            base = Image.new("RGBA", (280, 280), (0, 0, 0, 255))
            # keep inside 280x280, centered
            ratio = min(280 / pil.width, 280 / pil.height, 1.0)
            nw, nh = max(1, int(pil.width * ratio)), max(1, int(pil.height * ratio))
            pil = pil.resize((nw, nh))
            ox, oy = (280 - nw) // 2, (280 - nh) // 2
            base.paste(pil, (ox, oy), pil if pil.mode == "RGBA" else None)
            input_img = np.array(base)
        except Exception as e:
            st.error(f"Could not read image: {e}")

if input_img is None:
    st.info("Draw on the canvas (or upload an image) to get predictions.")
    st.stop()

# ================= Utilities (no scikit-image) =================
def otsu_threshold(arr_0_1: np.ndarray) -> float:
    """Otsu threshold on [0,1] image using NumPy only."""
    hist, _ = np.histogram((arr_0_1 * 255).astype(np.uint8), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = arr_0_1.size
    prob = hist / (total + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k_star = int(np.nanargmax(sigma_b2))
    return (k_star + 0.5) / 255.0

def resize_pil(arr_0_1: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize with Pillow (bilinear), keep [0,1] float32."""
    im = Image.fromarray((arr_0_1 * 255).astype("uint8"))
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

# ================= MNIST-style preprocessing =================
def preprocess(img_rgba):
    """
    1) grayscale [0..1]
    2) infer background from border; invert if bg is light
    3) Otsu binarize
    4) crop tight bbox
    5) resize longest side to 20 (keep aspect)
    6) center on 28x28 & normalize
    """
    img = Image.fromarray(img_rgba.astype(np.uint8)).convert("L")
    arr = np.array(img).astype("float32") / 255.0

    # background from 4px border
    b = 4
    border = np.concatenate([
        arr[:b, :].ravel(), arr[-b:, :].ravel(),
        arr[:, :b].ravel(), arr[:, -b:].ravel()
    ])
    if border.mean() > 0.5:
        arr = 1.0 - arr  # make digit white on black

    # Otsu threshold (NumPy)
    try:
        thr = otsu_threshold(arr)
    except Exception:
        thr = 0.3
    bin_img = (arr > thr).astype(np.uint8)

    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        arr28 = np.zeros((28, 28), dtype="float32")
        return arr28, arr28.reshape(1, 28, 28, 1), arr28.reshape(1, -1)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = resize_pil(cropped, new_h, new_w)

    arr28 = np.zeros((28, 28), dtype="float32")
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    arr28[y_off:y_off + new_h, x_off:x_off + new_w] = small

    m = arr28.max()
    if m > 0:
        arr28 = arr28 / m

    return arr28, arr28.reshape(1, 28, 28, 1), arr28.reshape(1, -1)

# ================= Predict helpers =================
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def _as_probs(vec):
    v = np.asarray(vec).reshape(-1)
    if v.size == 10:
        if np.allclose(np.sum(v), 1.0, rtol=1e-3, atol=1e-3):
            return v
        return _softmax(v)
    # if model outputs a class index
    oh = np.zeros(10, dtype="float32")
    oh[int(v[0])] = 1.0
    return oh

def keras_predict_any(model, arr4d, arr_flat):
    # Try NHWC first, then flat
    try:
        p = model.predict(arr4d, verbose=0)[0]
        return _as_probs(p)
    except Exception:
        p = model.predict(arr_flat, verbose=0)[0]
        return _as_probs(p)

def predict_dnn(arr4d, arr_flat):
    return keras_predict_any(dnn, arr4d, arr_flat)

def predict_cnn(arr4d, arr_flat):
    """If multiple CNNs loaded, average their probabilities."""
    if len(cnn_models) == 1:
        return keras_predict_any(cnn_models[0], arr4d, arr_flat)
    probs = np.zeros(10, dtype="float32")
    for m in cnn_models:
        probs += keras_predict_any(m, arr4d, arr_flat)
    return probs / float(len(cnn_models))

def predict_dt(arr_flat):
    try:
        return _as_probs(dt.predict_proba(arr_flat)[0])
    except Exception:
        cls = int(dt.predict(arr_flat)[0])
        oh = np.zeros(10, dtype="float32")
        oh[cls] = 1.0
        return oh

# ================= Inference =================
arr28, arr4d, arr_flat = preprocess(input_img[:, :, :3])

def predict_all():
    out = {}
    if dnn is not None:
        out["DNN"] = predict_dnn(arr4d, arr_flat)
    if len(cnn_models) > 0:
        label = "CNN" if len(cnn_models) == 1 else f"CNN (ensemble x{len(cnn_models)})"
        out[label] = predict_cnn(arr4d, arr_flat)
    if dt is not None:
        out["DecisionTree"] = predict_dt(arr_flat)
    return out

results = predict_all()

# ================= Display =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔎 What the model sees (28×28)")
    st.image((arr28 * 255).astype("uint8"), width=140, caption="Preprocessed 28×28 grayscale")
    st.caption("Tip: Draw thick strokes and keep the digit centered for best results.")

with col2:
    if mode == "Compare all":
        st.subheader("📊 Predictions (All Models)")
        for name, proba in results.items():
            pred = int(np.argmax(proba))
            conf = float(np.max(proba))
            st.write(f"**{name}** → Predicted: **{pred}**  | Confidence: {conf:.2f}")
            st.bar_chart(pd.Series(proba, index=list(range(10))))
    else:
        # chosen is one of ["DNN", "CNN", "DecisionTree"]
        if chosen == "CNN" and len(cnn_models) > 1:
            name = f"CNN (ensemble x{len(cnn_models)})"
        else:
            name = chosen
        st.subheader(f"📊 Prediction — {name}")
        # map chosen to vector
        if chosen == "DNN":
            proba = results[[k for k in results if k.startswith("DNN")][0]]
        elif chosen == "CNN":
            key = [k for k in results if k.startswith("CNN")][0]
            proba = results[key]
        else:
            proba = results["DecisionTree"]
        pred = int(np.argmax(proba))
        conf = float(np.max(proba))
        st.metric("Predicted Digit", pred, help="Highest-probability class")
        st.bar_chart(pd.Series(proba, index=list(range(10))))
