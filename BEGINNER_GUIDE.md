# 🎓 Week 3 Beginner's Guide: FastAPI & REST API

This guide explains everything we did in **Week 3** in simple terms so you can understand and explain it to your professor!

---

## 📚 What Did We Do This Week?

**Simple Answer:** We took the trained XGBoost model from Week 2 and turned it into a **web service** — meaning anyone can send data to it over the internet and get a prediction back!

**Why is this important?**
- A model saved as a file (`.pkl`) is useless to others
- A REST API allows any app, website, or script to use our model
- This is how real companies deploy ML models in production

**Analogy:** In Week 2, we built a smart calculator. In Week 3, we gave it a telephone number so others can call it and ask questions!

---

## 🌐 What is a REST API?

**REST API** = a way for programs to talk to each other over the internet using simple HTTP requests.

**Analogy:** Think of it like ordering at a restaurant:
- You (client) **send a request** → "I want a prediction for this customer session"
- The waiter (API) **processes it** → runs the ML model
- You receive a **response** → "This customer will likely Purchase (88.49% confidence)"

### HTTP Methods We Used

| Method | What it does | Real-world analogy |
|--------|-------------|-------------------|
| `GET` | Read information | Asking "What's the menu?" |
| `POST` | Send data, get result | "Here's my order, give me food" |

---

## ⚡ What is FastAPI?

**FastAPI** is a Python framework (a ready-made toolkit) for building APIs quickly and professionally.

**Why FastAPI?**
- **Fast to build:** Very little code needed
- **Automatic documentation:** Creates a beautiful interactive UI for FREE (Swagger)
- **Automatic validation:** Checks if your input data is correct before running the model
- **Industry standard:** Used by Netflix, Uber, Microsoft

---

## 📁 What We Created

### Project Structure (new files)
```
api/
├── __init__.py       ← Makes the folder a Python package
├── schemas.py        ← Defines the shape of data in and out
├── model.py          ← Loads the XGBoost model and preprocesses input
└── main.py           ← The actual API with all endpoints
```

---

## 🔍 File Explanations (Simple)

### 1. `api/schemas.py` — Data Validation

**What it does:** Defines exactly what data the API accepts and returns.

**Analogy:** Like a form with required fields. If you leave something blank or enter the wrong type, you get an error before the model even runs!

**We defined:**
- **Input:** All 17 original features (Administrative pages, Month, VisitorType, etc.)
- **Output:** Prediction (0 or 1), human-readable label, probability scores

**Example of what Pydantic checks:**
```
BounceRates: must be a number between 0 and 1 ✅
Month: must be one of ["Jan","Feb",...,"Dec"] ✅
Weekend: must be true or false ✅
If you send "abc" for BounceRates → API returns error 422 ❌
```

**Technology:** **Pydantic** (the validation library FastAPI uses)

---

### 2. `api/model.py` — Model Loader & Preprocessor

**What it does:** Loads the XGBoost model files and converts raw input into what the model expects.

**Why is this tricky?**
The model was trained on processed data (encoded months, scaled numbers). When someone sends us raw data like `Month: "Nov"`, we need to convert it to `Month_Nov: 1` exactly the way we did during training.

**Three steps:**
1. Load `xgboost.pkl` + `scaler.pkl` + `feature_names.pkl` from disk (done **once** at startup)
2. Encode input: convert `"Nov"` → `Month_Nov: 1`, convert `"Returning_Visitor"` → `VisitorType_Returning_Visitor: 1`
3. Scale: apply the same StandardScaler from Week 2
4. Run `.predict()` and `.predict_proba()`

**Key detail:** We load models ONCE at startup (not on every request) for performance.

---

### 3. `api/main.py` — The API Itself

**What it does:** Defines the 4 endpoints (URLs) the API responds to.

---

## 📡 The 4 Endpoints Explained

### Endpoint 1: `GET /` — Health Check

**What it does:** Just confirms the API is alive and the model loaded correctly.

**Like:** Picking up the phone and saying "Hello?" to confirm the line works.

```json
Response:
{
  "status": "ok",
  "message": "Online Shopper Purchase Intention API is running ✅",
  "model_loaded": true
}
```

---

### Endpoint 2: `GET /model-info` — Model Information

**What it does:** Returns information about the deployed model, including its Week 2 performance metrics.

**Like:** Asking the restaurant "What's your chef's specialty and how many stars do you have?"

```json
Response:
{
  "model_name": "XGBoost Classifier",
  "week2_metrics": {
    "accuracy": 0.8933,
    "f1_score": 0.658,
    "roc_auc": 0.928
  }
}
```

---

### Endpoint 3: `POST /predict` — Single Prediction

**What it does:** Takes one shopper session, runs it through XGBoost, returns a prediction.

**This is the main endpoint!**

**Example — A high-value customer:**
```json
Input (Request Body):
{
  "ProductRelated": 35,
  "ProductRelated_Duration": 2500.0,
  "PageValues": 25.4,
  "BounceRates": 0.01,
  "Month": "Nov",
  "VisitorType": "Returning_Visitor",
  "Weekend": false
  ... (other fields)
}

Output (Response):
{
  "prediction": 1,
  "label": "Purchase",
  "purchase_probability": 0.8849,
  "no_purchase_probability": 0.1151
}
```

**Interpretation:** This returning visitor browsed 35 product pages with high page value → **88.49% chance of purchase** ✅

---

### Endpoint 4: `POST /predict-batch` — Batch Predictions

**What it does:** Same as `/predict` but accepts a **list** of sessions at once.

**Why useful?** A company might want to score 10,000 customers at midnight without making 10,000 separate API calls.

```json
Input: [ session1, session2 ]

Output:
{
  "total": 2,
  "predictions": [
    { "label": "Purchase",    "purchase_probability": 0.8849 },
    { "label": "No Purchase", "purchase_probability": 0.0114 }
  ]
}
```

**Interpretation:**
- Session 1 (35 product pages, high value, November) → **Purchase** ✅
- Session 2 (5 product pages, PageValue=0, new visitor, high bounce rate) → **No Purchase** (98.86% confidence) ✅

---

## 🎨 Swagger UI — The Interactive Documentation

**What is Swagger?**
FastAPI **automatically creates** a beautiful web page at `http://localhost:8000/docs` where you can:
- See all endpoints with descriptions
- Click "Try it out" and test the API directly in the browser
- See example inputs and outputs
- No extra code needed — it's 100% automatic!

**How to open it:**
1. Run `uvicorn api.main:app --reload` in terminal
2. Open browser → `http://localhost:8000/docs`
3. Click any endpoint → "Try it out" → "Execute"

---

## 🔄 How the Full Flow Works (Step by Step)

```
Client (Swagger / App / Script)
        │
        │  POST /predict  {"ProductRelated": 35, "Month": "Nov", ...}
        ▼
FastAPI receives request
        │
        │  Pydantic validates: are all fields correct types?
        ▼
api/model.py preprocesses input:
  1. Encode Month      → Month_Nov = 1, others = 0
  2. Encode Visitor    → VisitorType_Returning_Visitor = 1
  3. Scale numbers     → StandardScaler.transform()
        │
        ▼
XGBoost model.predict() → 1
XGBoost model.predict_proba() → [0.1151, 0.8849]
        │
        ▼
FastAPI returns JSON response:
  { "prediction": 1, "label": "Purchase", "purchase_probability": 0.8849 }
        │
        ▼
Client receives result ✅
```

---

## 🛠️ Technologies Summary

| Technology | What it is | Why we used it |
|-----------|-----------|---------------|
| **FastAPI** | Python web framework | Build APIs quickly with auto-docs |
| **Uvicorn** | Web server | Runs the FastAPI application |
| **Pydantic** | Data validation library | Ensures inputs are correct before prediction |
| **Swagger UI** | Auto-generated web UI | Interactive API testing (free with FastAPI) |

---

## 🎯 How to Explain This to Your Professor

### Opening Statement
> "In Week 3, I deployed the XGBoost model from Week 2 as a REST API using FastAPI. The API exposes 4 endpoints that accept shopper session data and return purchase predictions in real time."

### Key Points to Make
1. **"I used FastAPI because it automatically generates Swagger documentation"** — show `/docs` in browser
2. **"Input validation is handled by Pydantic"** — prevents incorrect data from reaching the model
3. **"The preprocessing pipeline replicates exactly what was done during training"** — no data leakage, consistent results
4. **"The model loads once at startup for performance"** — not reloaded on every request
5. **"The batch endpoint allows scoring multiple customers at once"** — scalable design

### Live Demo (2 minutes)
1. `uvicorn api.main:app --reload` → show startup message
2. Open `http://localhost:8000/docs` → show the Swagger UI
3. Test `GET /` → show `status: ok, model_loaded: true`
4. Test `POST /predict` → use the pre-filled example → show Purchase (88.49%)

---

## 💡 Questions Your Professor Might Ask

**Q: "What is the difference between GET and POST?"**
**A:** "GET is for reading information without sending data — like checking the menu. POST is for sending data to get a result — like placing an order."

**Q: "Why did you use FastAPI instead of Flask?"**
**A:** "FastAPI automatically generates interactive API documentation via Swagger UI, has built-in data validation with Pydantic, and is significantly faster than Flask. It's also the industry standard for ML APIs."

**Q: "What happens if someone sends invalid data?"**
**A:** "Pydantic automatically validates every field before the data reaches the model. If BounceRates is not between 0 and 1, or if Month is not a valid month, FastAPI returns a 422 error with a clear explanation — without crashing."

**Q: "How do you make sure the preprocessing matches training?"**
**A:** "I use the exact same scaler (.pkl file) that was fitted during training. I also replicate the same encoding logic — same reference categories dropped for Month (August) and VisitorType (New_Visitor). This ensures the model receives data in exactly the same format it was trained on."

**Q: "What is Swagger?"**
**A:** "Swagger UI is an auto-generated interactive web page that documents all API endpoints. FastAPI creates it automatically from the Pydantic schemas and endpoint definitions — no extra code needed."

---

## ✅ Week 3 Deliverables Checklist

- ✅ **FastAPI environment** — installed, configured, running on port 8000
- ✅ **Pydantic schemas** — all 17 input features validated, structured output
- ✅ **Model loader** — XGBoost + scaler loaded once at startup
- ✅ **Preprocessing pipeline** — mirrors training exactly
- ✅ **`GET /`** — health check working
- ✅ **`GET /model-info`** — model metrics accessible
- ✅ **`POST /predict`** — single prediction working (tested: 88.49% Purchase)
- ✅ **`POST /predict-batch`** — batch prediction working (tested with 2 sessions)
- ✅ **Swagger UI** — available at `http://localhost:8000/docs`
- ✅ **README updated** — Week 3 progress, endpoint table, example JSON
- ✅ **requirements.txt updated** — fastapi, uvicorn, pydantic added

---

**Great job completing Week 3! Your model is now accessible as a professional REST API.** 🚀

---

**Week 3 completed:** February 2026
