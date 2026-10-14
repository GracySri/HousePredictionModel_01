import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

os.makedirs('Model', exist_ok=True)
os.makedirs('Documentation', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv('Dataset/train.csv')
print(f"   Shape: {df.shape}")

# ── Preprocessing ─────────────────────────────────────────────
print("\n🔧 Preprocessing...")

# Drop columns with >45% missing
missing_pct = df.isnull().mean()
df.drop(columns=missing_pct[missing_pct > 0.45].index, inplace=True)

# Separate numerical and categorical
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Impute using SimpleImputer (handles edge cases better)
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Outlier removal on SalePrice
Q1, Q3 = df['SalePrice'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['SalePrice'] >= Q1 - 1.5*IQR) & (df['SalePrice'] <= Q3 + 1.5*IQR)]
print(f"   Clean shape: {df.shape}")

# ── Feature Engineering ───────────────────────────────────────
print("\n⚙️  Feature engineering...")
df['HouseAge']  = df['YrSold'] - df['YearBuilt']
df['RemodAge']  = df['YrSold'] - df['YearRemodAdd']
df['TotalSF']   = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

# Encode categoricals
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Final impute pass (safety net for engineered features)
all_num = df.select_dtypes(include=np.number).columns
df[all_num] = SimpleImputer(strategy='median').fit_transform(df[all_num])

assert df.isnull().sum().sum() == 0, "Still have NaN!"

# Top 20 features by correlation
corr = df.corr()['SalePrice'].abs().sort_values(ascending=False)
selected_features = corr[1:21].index.tolist()
print(f"   Top features: {selected_features[:5]} ...")

# ── EDA Plots ─────────────────────────────────────────────────
print("\n📊 Saving EDA charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['SalePrice'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('SalePrice Distribution')
axes[1].hist(np.log1p(df['SalePrice']), bins=50, color='coral', edgecolor='white')
axes[1].set_title('Log(SalePrice) Distribution')
plt.tight_layout(); plt.savefig('Documentation/saleprice_distribution.png', dpi=120); plt.close()

top_heat = corr[1:11].index.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_heat + ['SalePrice']].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap'); plt.tight_layout()
plt.savefig('Documentation/correlation_heatmap.png', dpi=120); plt.close()

plt.figure(figsize=(10, 5))
df.boxplot(column='SalePrice', by='OverallQual', grid=False)
plt.title('SalePrice by Overall Quality'); plt.suptitle(''); plt.tight_layout()
plt.savefig('Documentation/boxplot_quality.png', dpi=120); plt.close()
print("   Charts saved!")

# ── Modelling ─────────────────────────────────────────────────
print("\n🤖 Training models...")
X = df[selected_features]
y = np.log1p(df['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

results = {}
def evaluate(name, yt, yp):
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    print(f"   {name:<25} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    results[name] = (mae, rmse, r2)

lr = LinearRegression().fit(Xs_train, y_train)
evaluate("Linear Regression", y_test, lr.predict(Xs_test))

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test, xgb.predict(X_test))

# ── Charts ────────────────────────────────────────────────────
names = list(results.keys())
r2s   = [results[m][2] for m in names]
rmses = [results[m][1] for m in names]
colors = ['#e74c3c','#f39c12','#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(names, r2s,   color=colors, edgecolor='white'); axes[0].set_title('R² Score (higher=better)'); axes[0].set_ylim(0,1)
axes[1].bar(names, rmses, color=colors, edgecolor='white'); axes[1].set_title('RMSE (lower=better)')
plt.tight_layout(); plt.savefig('Documentation/model_comparison.png', dpi=120); plt.close()

imp = pd.Series(rf.feature_importances_, index=selected_features).sort_values()
plt.figure(figsize=(9, 6)); imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest'); plt.tight_layout()
plt.savefig('Documentation/feature_importance.png', dpi=120); plt.close()

# ── Save ──────────────────────────────────────────────────────
print("\n💾 Saving model...")
pickle.dump(xgb,               open('Model/xgboost_model.pkl', 'wb'))
pickle.dump(scaler,            open('Model/scaler.pkl', 'wb'))
pickle.dump(selected_features, open('Model/features.pkl', 'wb'))

print("\n✅ ALL DONE!")
print("   Model  → Model/xgboost_model.pkl")
print("   Charts → Documentation/")
print(f"\n   Best Model R² = {results['XGBoost'][2]:.4f}")
print("\n🚀 Now run:  cd Streamlit_App && streamlit run app.py")
