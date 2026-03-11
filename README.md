# 💧 AI-Based Water Quality Prediction System

**Logistic Regression + Explainable AI + Claude-Powered Report Generation**

---

## Project Structure

```
water-quality-ai/
├── dataset/
│   └── water_potability.csv        ← put your dataset here
├── model/
│   ├── train_model.py              ← training pipeline
│   ├── water_model.pkl             ← saved after training
│   ├── scaler.pkl                  ← saved after training
│   └── plots/                      ← confusion matrix, ROC curve, feature importance
├── utils/
│   ├── risk_scorer.py              ← WHO parameter analysis + risk scoring
│   ├── report_generator.py         ← Claude API report + assistant
│   └── pdf_export.py               ← PDF download
├── app/
│   └── app.py                      ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Get the Dataset

Download the **Water Potability Dataset** from Kaggle:  
https://www.kaggle.com/datasets/adityakadiwal/water-potability

Save it as:
```
dataset/water_potability.csv
```

### Step 3 — Train the Model

```bash
python model/train_model.py
```

This will:
- Load and preprocess the dataset
- Train Logistic Regression
- Print accuracy, precision, recall, F1, AUC
- Save `water_model.pkl` and `scaler.pkl`
- Save plots: confusion matrix, ROC curve, feature importance

### Step 4 — Run the App

```bash
cd app
streamlit run app.py
```

Or from the project root:
```bash
streamlit run app/app.py
```

### Step 5 — Use the App

1. Open the browser (Streamlit opens it automatically at `localhost:8501`)
2. Enter your **Anthropic API Key** in the sidebar
3. Enter water parameters (or use the sample buttons)
4. Click **Predict Water Quality**
5. View: prediction, risk score, parameter charts, risk gauge, feature importance
6. Click **Generate AI Report** for Claude's analysis
7. Download the PDF report
8. Use the **AI Assistant** to ask questions about the water sample

---

## ML Concepts Used

| Concept | Where |
|---|---|
| Supervised Learning | Training on labeled water dataset |
| Binary Classification | Potable vs Not Potable |
| Logistic Regression | Core prediction model |
| Feature Engineering | 9 water quality parameters |
| Data Preprocessing | Missing value handling, StandardScaler |
| Train-Test Split | 80/20 stratified split |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, AUC-ROC |
| Feature Importance | Model coefficients visualization |
| Natural Language Generation | Claude API report generation |

---

## Features

- ✅ Water potability prediction with confidence score
- ✅ Water Quality Risk Score (0–100) with risk level
- ✅ WHO-standard parameter safety analysis
- ✅ Interactive parameter vs safe-limit bar chart
- ✅ Risk gauge visualization
- ✅ Feature importance chart
- ✅ Claude-powered AI water quality report
- ✅ Downloadable PDF report
- ✅ AI Water Quality Assistant (chat interface)
- ✅ Quick question buttons for the assistant

---

## Technologies

- Python
- scikit-learn (Logistic Regression)
- Streamlit (web interface)
- Plotly (visualizations)
- Anthropic Claude API (report + assistant)
- fpdf2 (PDF generation)
- SHAP (optional explainability)
