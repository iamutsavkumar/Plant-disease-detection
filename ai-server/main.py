"""
PlantMD – FastAPI AI Inference Server
Serves TensorFlow plant disease predictions on /predict
"""

import io, base64, logging, time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Optional TF import (gracefully degrades to mock in dev) ──────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not installed – running in MOCK mode")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plantmd")

# ─── PlantVillage class labels (38 classes) ───────────────────────────────────
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ─── Disease knowledge base ───────────────────────────────────────────────────
DISEASE_INFO: dict[str, dict] = {
    "Apple___Apple_scab": {
        "displayName": "Apple Scab",
        "plantType": "Apple",
        "severity": "Moderate",
        "urgency": "Medium",
        "symptoms": [
            "Olive-green to brown velvety lesions on leaves",
            "Scabby, cracked spots on fruit surface",
            "Premature leaf and fruit drop",
            "Distorted, stunted fruit development",
        ],
        "causes": [
            "Fungal pathogen Venturia inaequalis",
            "Thrives in cool, moist spring weather (10–24 °C)",
            "Overwinters in infected leaf debris on soil surface",
        ],
        "treatment": [
            "Apply fungicide (captan, myclobutanil) at green-tip stage",
            "Repeat sprays every 7–10 days during wet periods",
            "Remove and destroy fallen infected leaves",
            "Prune for better air circulation inside canopy",
        ],
        "prevention": [
            "Plant scab-resistant cultivars (Enterprise, Liberty, Redfree)",
            "Rake and destroy leaf litter in autumn",
            "Avoid overhead irrigation; use drip systems",
            "Monitor weather: apply protectant before rain events",
        ],
        "additionalNotes": "Economic damage threshold is reached at ~10% leaf infection. Act early in spring.",
    },
    "Apple___Black_rot": {
        "displayName": "Apple Black Rot",
        "plantType": "Apple",
        "severity": "Severe",
        "urgency": "High",
        "symptoms": [
            "Brown 'frogeye' leaf spots with purple margins",
            "Black, mummified fruit remaining on tree",
            "Reddish-brown cankers on branches",
            "Fruit turns black and shrivels",
        ],
        "causes": [
            "Fungal pathogen Botryosphaeria obtusa",
            "Enters through wounds, pruning cuts, or fire blight lesions",
            "Spreads via rain splash and wind",
        ],
        "treatment": [
            "Prune out infected wood 15–30 cm beyond visible canker",
            "Remove mummified fruit from tree and ground",
            "Apply copper-based fungicide in early spring",
            "Use captan or thiophanate-methyl during bloom period",
        ],
        "prevention": [
            "Avoid wounding trees during harvest and pruning",
            "Disinfect pruning tools with 10% bleach solution",
            "Control fire blight and other stress factors",
            "Ensure adequate tree nutrition to maintain vigor",
        ],
        "additionalNotes": "Often secondary to tree stress. Improve overall orchard management first.",
    },
    "Tomato___Early_blight": {
        "displayName": "Tomato Early Blight",
        "plantType": "Tomato",
        "severity": "Moderate",
        "urgency": "Medium",
        "symptoms": [
            "Dark concentric 'target board' rings on lower leaves",
            "Yellow halo surrounding brown lesions",
            "Lesions enlarge and leaves yellow, wither, and drop",
            "Dark stem lesions near soil level (collar rot)",
        ],
        "causes": [
            "Fungal pathogen Alternaria solani",
            "Favored by warm temperatures 24–29 °C and high humidity",
            "Survives in soil and infected plant debris",
        ],
        "treatment": [
            "Remove and destroy infected lower leaves promptly",
            "Apply chlorothalonil, mancozeb, or copper fungicide",
            "Spray on 7–10 day schedule during humid weather",
            "Stake plants to improve air circulation",
        ],
        "prevention": [
            "Rotate crops on 3-year cycle; avoid Solanaceae family",
            "Mulch to prevent soil splash onto leaves",
            "Water at soil level in the morning",
            "Plant resistant varieties (Iron Lady, Defiant)",
        ],
        "additionalNotes": "Begins on oldest, lowest leaves. Regular scouting is essential.",
    },
    "Tomato___Late_blight": {
        "displayName": "Tomato Late Blight",
        "plantType": "Tomato",
        "severity": "Severe",
        "urgency": "Critical",
        "symptoms": [
            "Large, water-soaked greenish-grey leaf patches",
            "White fuzzy sporulation on leaf undersides in humidity",
            "Dark brown lesions spread rapidly across leaves",
            "Firm brown rot on fruits, often with greasy appearance",
        ],
        "causes": [
            "Oomycete pathogen Phytophthora infestans",
            "Thrives in cool 10–20 °C temperatures with wet conditions",
            "Spreads explosively via air-dispersed sporangia",
        ],
        "treatment": [
            "Apply protectant fungicide (chlorothalonil) immediately",
            "Use systemic fungicide (metalaxyl, cymoxanil) for active infection",
            "Remove and bag all infected tissue; do NOT compost",
            "Destroy entire plant if >30% canopy affected",
        ],
        "prevention": [
            "Use certified disease-free seed and transplants",
            "Avoid overhead irrigation entirely",
            "Plant resistant varieties (Defiant PhR, Mountain Magic)",
            "Scout daily during cool, wet weather periods",
        ],
        "additionalNotes": "⚠ This is the same pathogen that caused the Irish Potato Famine. Act within 24 hours.",
    },
    "Corn_(maize)___Common_rust_": {
        "displayName": "Corn Common Rust",
        "plantType": "Corn (Maize)",
        "severity": "Moderate",
        "urgency": "Medium",
        "symptoms": [
            "Small, oval cinnamon-brown pustules on both leaf surfaces",
            "Pustules rupture releasing powdery reddish-brown spores",
            "Leaves turn yellow then die from heavy infection",
            "Husks and sheaths may also be affected",
        ],
        "causes": [
            "Fungal pathogen Puccinia sorghi",
            "Spreads via wind-blown urediniospores",
            "Favors cool 16–23 °C temperatures with high humidity",
        ],
        "treatment": [
            "Apply triazole fungicide (propiconazole) at first sign",
            "Foliar application of strobilurin fungicides",
            "Treat before tasseling for best protection of yield",
        ],
        "prevention": [
            "Plant rust-resistant hybrids suited to your region",
            "Avoid late planting that extends exposure season",
            "Monitor fields from V6 stage through silking",
        ],
        "additionalNotes": "Economic threshold: >5% leaf area affected before tasseling warrants treatment.",
    },
    "Grape___Black_rot": {
        "displayName": "Grape Black Rot",
        "plantType": "Grape",
        "severity": "Severe",
        "urgency": "High",
        "symptoms": [
            "Tan-brown circular leaf lesions with dark border",
            "Black fungal pycnidia visible in lesion centers",
            "Berries shrivel into hard black 'mummies'",
            "Young shoots develop reddish-brown lesions",
        ],
        "causes": [
            "Fungal pathogen Guignardia bidwellii",
            "Overwinters in mummified berries and infected canes",
            "Primary infections from ascospores during bloom period",
        ],
        "treatment": [
            "Apply mancozeb or myclobutanil from bud break through veraison",
            "Remove all mummified berries before budbreak",
            "Prune heavily infected canes during dormancy",
        ],
        "prevention": [
            "Train vines for maximum air circulation",
            "Remove mummies and debris from vineyard floor annually",
            "Apply dormant-season copper sprays",
        ],
        "additionalNotes": "Most critical infection window is 2–4 weeks after bloom. Protect berries early.",
    },
    "Potato___Late_blight": {
        "displayName": "Potato Late Blight",
        "plantType": "Potato",
        "severity": "Severe",
        "urgency": "Critical",
        "symptoms": [
            "Pale green, water-soaked spots on leaf margins",
            "Lesions turn dark brown/black and expand rapidly",
            "White sporulation on leaf undersides in humid conditions",
            "Tubers show reddish-brown internal rot",
        ],
        "causes": [
            "Oomycete pathogen Phytophthora infestans",
            "Spreads through infected seed tubers and air-dispersed spores",
            "Epidemics occur under cool, wet blight-favorable conditions",
        ],
        "treatment": [
            "Apply protectant fungicides preventatively (chlorothalonil, mancozeb)",
            "Use systemic fungicides (fluopicolide, cymoxanil) when symptoms appear",
            "Destroy infected haulm before harvest",
            "Do not harvest during wet conditions to avoid tuber infection",
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Use blight-resistant varieties (Sarpo Mira, Cara)",
            "Hill soil over tubers to protect from rain-washed spores",
            "Monitor using online blight forecasting services",
        ],
        "additionalNotes": "Can devastate entire crop within 7–10 days under ideal conditions for the pathogen.",
    },
    "_healthy": {
        "displayName": "Healthy Plant",
        "plantType": "Unknown",
        "severity": "None",
        "urgency": "Low",
        "symptoms": ["No visible disease symptoms detected", "Leaf color appears normal for species"],
        "causes": [],
        "treatment": ["No immediate treatment required"],
        "prevention": [
            "Maintain regular watering schedule appropriate to species",
            "Ensure adequate nutrition with balanced fertilizer",
            "Monitor periodically for early signs of stress or disease",
            "Promote air circulation and good sanitation in growing area",
        ],
        "additionalNotes": "Plant appears healthy. Continue current care routine.",
    },
}

def get_disease_info(label: str) -> dict:
    """Return knowledge base entry for a label, with sensible defaults."""
    if label in DISEASE_INFO:
        return DISEASE_INFO[label]
    # Try suffix match for healthy classes
    if label.endswith("___healthy"):
        base = DISEASE_INFO["_healthy"].copy()
        plant = label.split("___")[0].replace("_", " ").title()
        base["plantType"] = plant
        base["displayName"] = f"Healthy {plant}"
        return base
    # Generic fallback for unknown diseases
    parts = label.split("___")
    plant = parts[0].replace("_", " ").replace(",", "").title()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown Disease"
    return {
        "displayName": disease,
        "plantType": plant,
        "severity": "Moderate",
        "urgency": "Medium",
        "symptoms": ["Visible lesions or discoloration on leaf tissue", "Abnormal growth pattern detected"],
        "causes": ["Pathogen infection (fungal, bacterial, or viral)", "Environmental stress factors may contribute"],
        "treatment": [
            "Isolate affected plants to prevent spread",
            "Consult local agricultural extension service",
            "Apply broad-spectrum fungicide as interim measure",
        ],
        "prevention": [
            "Maintain good sanitation practices in growing area",
            "Rotate crops and avoid overwatering",
            "Scout regularly for early detection",
        ],
        "additionalNotes": f"Detected: {disease} on {plant}. Consult a local agronomist for confirmation.",
    }


# ─── Model singleton ──────────────────────────────────────────────────────────
model: "tf.keras.Model | None" = None
IMG_SIZE = 224


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = Path("model/plantmd_model.keras")
    if TF_AVAILABLE and model_path.exists():
        log.info("Loading TensorFlow model from %s", model_path)
        model = tf.keras.models.load_model(str(model_path))
        log.info("Model loaded. Input shape: %s", model.input_shape)
    else:
        log.warning("Model not found at %s — using MOCK predictions", model_path)
    yield
    log.info("Shutting down AI server")


app = FastAPI(title="PlantMD AI Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image: str          # base64-encoded image (with or without data URI prefix)
    mediaType: str = "image/jpeg"


class PredictResponse(BaseModel):
    isPlant: bool
    isHealthy: bool
    diseaseName: str
    plantType: str
    confidence: int
    severity: str
    symptoms: list[str]
    causes: list[str]
    treatment: list[str]
    prevention: list[str]
    urgency: str
    additionalNotes: str
    topPredictions: list[dict]
    inferenceMs: int


# ─── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(b64: str) -> "np.ndarray":
    """Decode base64 → PIL → numpy array normalised to [0,1], shape (1, 224, 224, 3)."""
    # Strip data URI prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)        # (1, 224, 224, 3)


def is_plant_image(arr: "np.ndarray") -> bool:
    """
    Heuristic: plant leaves are predominantly green.
    Check that mean green channel > red and blue channels by threshold.
    """
    img = arr[0]                          # (224, 224, 3)
    r, g, b = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    return g > (r - 0.05) and g > (b - 0.05) and g > 0.12


# ─── Mock prediction (development without a trained model) ────────────────────
def mock_predict(arr: "np.ndarray") -> tuple[str, int, list[dict]]:
    rng = np.random.default_rng(seed=int(arr.mean() * 1e6))
    idx = int(rng.integers(0, len(CLASS_LABELS)))
    conf = int(rng.integers(72, 97))
    label = CLASS_LABELS[idx]
    others = rng.choice([i for i in range(len(CLASS_LABELS)) if i != idx], 4, replace=False)
    top = [{"label": label, "confidence": conf}] + [
        {"label": CLASS_LABELS[i], "confidence": int(rng.integers(10, 40))} for i in others
    ]
    return label, conf, top


# ─── Real TF prediction ───────────────────────────────────────────────────────
def tf_predict(arr: "np.ndarray") -> tuple[str, int, list[dict]]:
    preds = model.predict(arr, verbose=0)[0]     # (38,)
    top5_idx = preds.argsort()[-5:][::-1]
    label = CLASS_LABELS[top5_idx[0]]
    conf = int(round(float(preds[top5_idx[0]]) * 100))
    top = [{"label": CLASS_LABELS[i], "confidence": int(round(float(preds[i]) * 100))} for i in top5_idx]
    return label, conf, top


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tf_available": TF_AVAILABLE,
        "classes": len(CLASS_LABELS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        arr = preprocess_image(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {exc}")

    if not is_plant_image(arr):
        return PredictResponse(
            isPlant=False,
            isHealthy=False,
            diseaseName="Not a Plant Image",
            plantType="N/A",
            confidence=95,
            severity="None",
            symptoms=["No plant leaf detected in the image"],
            causes=["Image does not appear to contain plant foliage"],
            treatment=["Please upload a clear photo of a plant leaf"],
            prevention=[],
            urgency="Low",
            additionalNotes="Upload a well-lit, close-up photo of a leaf for accurate diagnosis.",
            topPredictions=[],
            inferenceMs=int((time.perf_counter() - t0) * 1000),
        )

    if model is not None and TF_AVAILABLE:
        label, conf, top = tf_predict(arr)
    else:
        label, conf, top = mock_predict(arr)

    info = get_disease_info(label)
    is_healthy = "healthy" in label.lower()

    return PredictResponse(
        isPlant=True,
        isHealthy=is_healthy,
        diseaseName=info["displayName"],
        plantType=info["plantType"],
        confidence=conf,
        severity=info["severity"],
        symptoms=info["symptoms"],
        causes=info["causes"],
        treatment=info["treatment"],
        prevention=info["prevention"],
        urgency=info["urgency"],
        additionalNotes=info["additionalNotes"],
        topPredictions=top,
        inferenceMs=int((time.perf_counter() - t0) * 1000),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )