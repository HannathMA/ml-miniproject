import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

# Find the main project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_FILE = BASE_DIR / "data" / "poverty_data.csv"

# Read CSV file
df = pd.read_csv(DATA_FILE)


# =========================================================
# 2. DISPLAY BASIC INFORMATION
# =========================================================

print("\n========== DATASET LOADED SUCCESSFULLY ==========")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nORIGINAL DATASET SHAPE:")
print(df.shape)


# =========================================================
# 3. REMOVE EMPTY COLUMN
# =========================================================

df = df.drop(columns=["Unnamed: 22"], errors="ignore")

print("\nDATASET SHAPE AFTER REMOVING EMPTY COLUMN:")
print(df.shape)


# =========================================================
# 4. SELECT NATIONAL POVERTY INDICATOR
# =========================================================

df = df[
    df["Indicator Name"].str.contains(
        "national poverty",
        case=False,
        na=False
    )
].copy()

print("\n========== NATIONAL POVERTY DATA ==========")

print("\nDATASET SHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())


# =========================================================
# 5. SELECT TARGET COLUMN
# =========================================================

# We use poverty value in the year 2011 as the target
target = "2011"

print("\n========== TARGET ==========")
print("Target Column:", target)


# Check whether target column exists
if target not in df.columns:
    print(f"\nERROR: Target column '{target}' does not exist!")
    print("Available columns:")
    print(df.columns.tolist())
    exit()


# Remove rows where target value is missing
df = df.dropna(subset=[target])

print("\nDATASET SHAPE AFTER REMOVING MISSING TARGET VALUES:")
print(df.shape)


# =========================================================
# 6. SELECT FEATURE COLUMNS
# =========================================================

features = [
    "2007",
    "2008",
    "2009",
    "2010",
    "2012"
]

# Keep only columns that actually exist
features = [
    column
    for column in features
    if column in df.columns
]

print("\n========== FEATURES ==========")
print(features)


# =========================================================
# 7. PREPARE X AND y
# =========================================================

X = df[features].copy()

y = df[target].copy()


# =========================================================
# 8. HANDLE MISSING VALUES
# =========================================================

print("\nMISSING VALUES BEFORE CLEANING:")

print(X.isnull().sum())


# Fill missing values using column mean
X = X.fillna(X.mean())


# Remove columns that are completely empty
X = X.dropna(axis=1, how="all")


# Remove rows that still contain missing values
valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows]

y = y.loc[valid_rows]


print("\nMISSING VALUES AFTER CLEANING:")

print(X.isnull().sum())


print("\n========== FINAL DATA ==========")

print("FINAL FEATURE COLUMNS:")
print(X.columns.tolist())

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)


# =========================================================
# 9. CHECK IF ENOUGH DATA EXISTS
# =========================================================

if len(X) < 10:
    print("\nERROR: Not enough data available for training.")
    exit()


if X.shape[1] < 2:
    print("\nERROR: Not enough features available for PCA.")
    exit()


# =========================================================
# 10. SPLIT DATA INTO TRAINING AND TESTING
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========")

print("Training Samples:", len(X_train))

print("Testing Samples:", len(X_test))


# =========================================================
# 11. STANDARDIZE THE DATA
# =========================================================

scaler = StandardScaler()


# Fit scaler using training data
X_train_scaled = scaler.fit_transform(X_train)


# Transform testing data
X_test_scaled = scaler.transform(X_test)


print("\n========== DATA STANDARDIZATION COMPLETE ==========")


# =========================================================
# 12. APPLY PCA
# =========================================================

# Keep enough principal components to preserve 95% variance
pca = PCA(
    n_components=0.95
)


# Apply PCA to training data
X_train_pca = pca.fit_transform(
    X_train_scaled
)


# Apply same PCA transformation to testing data
X_test_pca = pca.transform(
    X_test_scaled
)


print("\n========== PCA RESULTS ==========")

print("\nORIGINAL NUMBER OF FEATURES:")

print(X_train.shape[1])


print("\nNUMBER OF PCA COMPONENTS:")

print(pca.n_components_)


print("\nEXPLAINED VARIANCE RATIO:")

print(pca.explained_variance_ratio_)


total_variance = (
    pca.explained_variance_ratio_.sum()
)


print("\nTOTAL EXPLAINED VARIANCE:")

print(
    f"{total_variance * 100:.2f}%"
)


# =========================================================
# 13. MODEL WITHOUT PCA
# =========================================================

print("\n========== MODEL WITHOUT PCA ==========")


model_original = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# Train model
model_original.fit(
    X_train_scaled,
    y_train
)


# Predict training data
y_train_pred_original = model_original.predict(
    X_train_scaled
)


# Predict testing data
y_test_pred_original = model_original.predict(
    X_test_scaled
)


# Training scores
train_mae_original = mean_absolute_error(
    y_train,
    y_train_pred_original
)

train_mse_original = mean_squared_error(
    y_train,
    y_train_pred_original
)

train_r2_original = r2_score(
    y_train,
    y_train_pred_original
)


# Testing scores
test_mae_original = mean_absolute_error(
    y_test,
    y_test_pred_original
)

test_mse_original = mean_squared_error(
    y_test,
    y_test_pred_original
)

test_r2_original = r2_score(
    y_test,
    y_test_pred_original
)


print("\nTRAINING RESULTS WITHOUT PCA")

print("MAE:", train_mae_original)

print("MSE:", train_mse_original)

print("R2 Score:", train_r2_original)


print("\nTESTING RESULTS WITHOUT PCA")

print("MAE:", test_mae_original)

print("MSE:", test_mse_original)

print("R2 Score:", test_r2_original)


# =========================================================
# 14. MODEL WITH PCA
# =========================================================

print("\n========== MODEL WITH PCA ==========")


model_pca = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# Train model using PCA components
model_pca.fit(
    X_train_pca,
    y_train
)


# Predict training data
y_train_pred_pca = model_pca.predict(
    X_train_pca
)


# Predict testing data
y_test_pred_pca = model_pca.predict(
    X_test_pca
)


# Training scores
train_mae_pca = mean_absolute_error(
    y_train,
    y_train_pred_pca
)

train_mse_pca = mean_squared_error(
    y_train,
    y_train_pred_pca
)

train_r2_pca = r2_score(
    y_train,
    y_train_pred_pca
)


# Testing scores
test_mae_pca = mean_absolute_error(
    y_test,
    y_test_pred_pca
)

test_mse_pca = mean_squared_error(
    y_test,
    y_test_pred_pca
)

test_r2_pca = r2_score(
    y_test,
    y_test_pred_pca
)


print("\nTRAINING RESULTS WITH PCA")

print("MAE:", train_mae_pca)

print("MSE:", train_mse_pca)

print("R2 Score:", train_r2_pca)


print("\nTESTING RESULTS WITH PCA")

print("MAE:", test_mae_pca)

print("MSE:", test_mse_pca)

print("R2 Score:", test_r2_pca)


# =========================================================
# 15. OVERFITTING CHECK
# =========================================================

print("\n========== OVERFITTING ANALYSIS ==========")


print("\nWITHOUT PCA:")

print("Training R2:", train_r2_original)

print("Testing R2:", test_r2_original)


if train_r2_original - test_r2_original > 0.15:
    print("Result: Possible Overfitting")
else:
    print("Result: Model performance is relatively balanced")


print("\nWITH PCA:")

print("Training R2:", train_r2_pca)

print("Testing R2:", test_r2_pca)


if train_r2_pca - test_r2_pca > 0.15:
    print("Result: Possible Overfitting")
else:
    print("Result: Model performance is relatively balanced")


# =========================================================
# 16. FINAL COMPARISON
# =========================================================

print("\n========== FINAL PCA COMPARISON ==========")

print("\nNumber of Original Features:")
print(X.shape[1])

print("\nNumber of PCA Components:")
print(pca.n_components_)

print("\nVariance Retained:")
print(f"{total_variance * 100:.2f}%")


print("\nMODEL PERFORMANCE COMPARISON")

print("\nWITHOUT PCA")
print("Testing MAE:", test_mae_original)
print("Testing MSE:", test_mse_original)
print("Testing R2:", test_r2_original)


print("\nWITH PCA")
print("Testing MAE:", test_mae_pca)
print("Testing MSE:", test_mse_pca)
print("Testing R2:", test_r2_pca)


print("\n========== PROGRAM COMPLETED SUCCESSFULLY ==========")