import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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
# SELECT POVERTY DATA
# -----------------------------------------

# Select only the national poverty indicator
df = df[
    df["Indicator Name"].str.contains(
        "national poverty",
        case=False,
        na=False
    )
]

print("\nDATA AFTER SELECTING NATIONAL POVERTY INDICATOR:")
print(df.shape)


# -----------------------------------------
# SELECT YEAR COLUMNS
# -----------------------------------------

features = [
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012"
]

# Keep only existing columns
features = [col for col in features if col in df.columns]

print("\nDBSCAN FEATURES:")
print(features)


# -----------------------------------------
# REMOVE EMPTY COLUMNS
# -----------------------------------------

X = df[features].copy()

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# Fill missing values with column mean
X = X.fillna(X.mean())

# Remove any remaining missing values
X = X.dropna()

# Keep corresponding rows in original dataframe
df = df.loc[X.index].copy()


# -----------------------------------------
# STANDARDIZE DATA
# -----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -----------------------------------------
# APPLY DBSCAN
# -----------------------------------------

dbscan = DBSCAN(
    eps=0.8,
    min_samples=3
)

df["Cluster"] = dbscan.fit_predict(X_scaled)


# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------

print("\n========== DBSCAN RESULTS ==========")

print("\nCLUSTER COUNTS:")
print(df["Cluster"].value_counts().sort_index())

print("\n-1 means NOISE / OUTLIER")


print("\nCOUNTRY CLUSTERS:")

print(
    df[
        [
            "Country Name",
            "Indicator Name",
            "Cluster"
        ]
    ].sort_values(
        by=["Cluster", "Country Name"]
    ).to_string(index=False)
)


# -----------------------------------------
# CLUSTER SUMMARY
# -----------------------------------------

print("\n========== CLUSTER SUMMARY ==========")

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    if cluster == -1:
        print(
            f"\nNoise / Outliers: "
            f"{len(cluster_data)} countries"
        )
    else:
        print(
            f"\nCluster {cluster}: "
            f"{len(cluster_data)} countries"
        )

        print(
            cluster_data[
                "Country Name"
            ].tolist()
        )