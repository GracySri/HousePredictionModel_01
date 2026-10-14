# 🏠 House Price Prediction System
### AIML Summer Internship 2026 — IIHMF, MNNIT Allahabad

---

## 📁 Folder Structure

```
HousePricePrediction/
│
├── Dataset/
│   └── train.csv               ← Download from Kaggle (link below)
│
├── Notebook/
│   └── house_price_prediction.ipynb   ← Main Jupyter Notebook
│
├── Model/
│   ├── xgboost_model.pkl       ← Saved after running notebook
│   ├── scaler.pkl
│   └── features.pkl
│
├── Streamlit_App/
│   └── app.py                  ← Deployment app
│
├── Documentation/
│   └── (charts auto-saved here by notebook)
│
└── README.md
```

---

## 📦 Dataset

Download from Kaggle:
🔗 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Place `train.csv` inside the `Dataset/` folder.

---

## ⚙️ Setup & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

---

## 🚀 How to Run

### Step 1 — Run the Notebook
```bash
cd Notebook/
jupyter notebook house_price_prediction.ipynb
```
Run all cells. This will:
- Preprocess data
- Train 3 models (Linear Regression, Random Forest, XGBoost)
- Save the best model to `Model/`

### Step 2 — Launch Streamlit App
```bash
cd Streamlit_App/
streamlit run app.py
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Ensemble |
| XGBoost Regressor | Boosting ✅ Best |

---

## 📊 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 👤 Submission Format
`HousePricePrediction_YourName.zip`
