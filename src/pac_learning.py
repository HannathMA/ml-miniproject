import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "poverty_data.csv"

df = pd.read_csv(DATA_FILE)

# Remove empty column
df = df.drop(columns=["Unnamed: 22"], errors="ignore")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())


# -----------------------------------------
# SELECT NUMERICAL COLUMNS
# -----------------------------------------

numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

print("\nNUMERICAL COLUMNS:")
print(numerical_columns)


# -----------------------------------------
# CREATE TARGET
# -----------------------------------------

target = "2011"

# Rename target column for clarity
df = df.rename(columns={target: "Poverty_Rate"})

target = "Poverty_Rate"

print("\nTARGET:")
print(target)


# -----------------------------------------
# REMOVE ROWS WHERE TARGET IS MISSING
# -----------------------------------------

df = df.dropna(subset=[target])

print("\nDATASET SHAPE AFTER REMOVING MISSING TARGET:")
print(df.shape)


# -----------------------------------------
# SELECT FEATURES
# -----------------------------------------

features = [
    "2007",
    "2008",
    "2009",
    "2010",
    "2012",
    "2013"
]

# Keep only columns that actually exist
features = [col for col in features if col in df.columns]

print("\nFEATURES:")
print(features)


# -----------------------------------------
# HANDLE MISSING FEATURE VALUES
# -----------------------------------------

X = df[features].copy()
y = df[target]

# Fill missing values with column mean
X = X.fillna(X.mean())

# Remove columns that are completely empty
X = X.dropna(axis=1, how="all")

print("\nFINAL FEATURES:")
print(X.columns.tolist())

print("\nX SHAPE:", X.shape)
print("y SHAPE:", y.shape)


# -----------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)
print("TESTING DATA SIZE:", X_test.shape)


# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------------------
# PREDICTIONS
# -----------------------------------------

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# -----------------------------------------
# EVALUATION
# -----------------------------------------

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n========== MODEL RESULTS ==========")

print("\nTRAINING RESULTS")
print("MAE:", train_mae)
print("R² Score:", train_r2)

print("\nTESTING RESULTS")
print("MAE:", test_mae)
print("R² Score:", test_r2)


# -----------------------------------------
# OVERFITTING CHECK
# -----------------------------------------

print("\n========== OVERFITTING CHECK ==========")

if train_r2 > test_r2 + 0.15:
    print("Possible overfitting detected.")
elif abs(train_r2 - test_r2) < 0.15:
    print("Model performance is relatively balanced.")
else:
    print("Model may be underfitting.")


# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))