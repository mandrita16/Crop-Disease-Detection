import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "../data/tabular/Crop_recommendation.csv"

# Ideal ranges per feature (agronomic reference midpoints - cite FAO/ICAR
# guidelines in your report for these values)
IDEAL_RANGES = {
    "N": (50, 100),
    "P": (30, 70),
    "K": (30, 80),
    "temperature": (20, 30),
    "humidity": (60, 85),
    "ph": (6.0, 7.5),
    "rainfall": (100, 200),
}


def compute_quality_score(row):
    """
    Deviation-based quality score: for each feature, penalize distance
    from the ideal range. Aggregate into a 0-100 quality score, then
    bucket into Good / Moderate / Poor.
    """
    penalties = []
    for feature, (low, high) in IDEAL_RANGES.items():
        val = row[feature]
        if val < low:
            penalty = (low - val) / low
        elif val > high:
            penalty = (val - high) / high
        else:
            penalty = 0.0
        penalties.append(penalty)

    avg_penalty = np.mean(penalties)
    score = max(0, 100 - avg_penalty * 100)
    return score


def bucket_quality(score):
    if score >= 75:
        return "Good"
    elif score >= 50:
        return "Moderate"
    else:
        return "Poor"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    df["quality_score"] = df.apply(compute_quality_score, axis=1)
    df["quality_label"] = df["quality_score"].apply(bucket_quality)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features]
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "../models/tabular_scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, features, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, df = load_and_prepare()
    print(df[["N", "P", "K", "quality_score", "quality_label"]].head(10))
    print("\nClass distribution:\n", df["quality_label"].value_counts())
