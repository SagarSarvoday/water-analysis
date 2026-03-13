# AI-Based Water Quality Prediction System

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Place `water_potability.csv` inside the `dataset/` folder
2. Train all models:
```bash
python model/train_model.py
```
3. Run the web app:
```bash
streamlit run app/app.py
```

## Project Structure
```
water-quality-ai/
├── dataset/
│   └── water_potability.csv        ← Add your dataset here
├── model/
│   ├── train_model.py              ← Training pipeline
│   ├── water_model.pkl             ← Generated after training
│   ├── scaler.pkl                  ← Generated after training
│   ├── logistic_regression.pkl     ← Generated after training
│   ├── random_forest.pkl           ← Generated after training
│   ├── gradient_boosting.pkl       ← Generated after training
│   ├── improved_logistic_regression.pkl
│   └── plots/                      ← All charts generated here
├── utils/
│   ├── risk_scorer.py
│   └── report_generator.py
├── app/
│   └── app.py
└── requirements.txt
```

## Faculty Requirements Covered
- Algorithm Comparison: LR vs Random Forest vs Gradient Boosting
- Improved Logistic Regression: SMOTE + RobustScaler + GridSearchCV + RFE
- ML-based Feature Selection: RF Importance + RFE
- AI Report Generation via Groq API
- Interactive Web App with Explainable AI
