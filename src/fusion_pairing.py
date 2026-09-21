import numpy as np
import pandas as pd
import json

np.random.seed(42)

# Rough mapping: disease severity level -> tabular quality tendency
# (documented assumption for your report's methodology section)
DISEASE_TO_QUALITY_BIAS = {
    "healthy": "Good",
    "mild": "Moderate",
    "severe": "Poor",
}


def classify_disease_severity(class_name):
    name = class_name.lower()
    if "healthy" in name:
        return "healthy"
    # crude heuristic: treat blight/rot/rust/wilt as severe, everything else mild
    severe_keywords = ["blight", "rot", "rust", "wilt"]
    if any(k in name for k in severe_keywords):
        return "severe"
    return "mild"


def generate_paired_dataset(class_names, tabular_df, n_pairs_per_class=200):
    """
    For each disease class, sample tabular rows biased toward the
    quality label that matches the disease's severity, to create a
    plausible joint dataset for fusion training.
    """
    rows = []
    for class_name in class_names:
        severity = classify_disease_severity(class_name)
        target_quality = DISEASE_TO_QUALITY_BIAS[severity]

        matching = tabular_df[tabular_df["quality_label"] == target_quality]
        # occasionally sample a mismatched row too, so the fusion model
        # actually learns to handle disagreement rather than always agreeing
        mismatched = tabular_df[tabular_df["quality_label"] != target_quality]

        n_match = int(n_pairs_per_class * 0.8)
        n_mismatch = n_pairs_per_class - n_match

        sampled_match = matching.sample(n=min(n_match, len(matching)), replace=True, random_state=42)
        sampled_mismatch = mismatched.sample(n=min(n_mismatch, len(mismatched)), replace=True, random_state=42)

        for _, row in pd.concat([sampled_match, sampled_mismatch]).iterrows():
            rows.append({
                "disease_class": class_name,
                "disease_severity": severity,
                "N": row["N"], "P": row["P"], "K": row["K"],
                "temperature": row["temperature"], "humidity": row["humidity"],
                "ph": row["ph"], "rainfall": row["rainfall"],
                "tabular_quality_label": row["quality_label"],
                # ground truth for fusion training: majority vote of the two signals
                "true_final_label": target_quality if row["quality_label"] == target_quality else "Moderate",
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from tabular_data import load_and_prepare

    with open("../models/class_names.json") as f:
        class_names = json.load(f)

    _, _, _, _, _, tabular_df = load_and_prepare()

    paired_df = generate_paired_dataset(class_names, tabular_df)
    paired_df.to_csv("../data/tabular/paired_fusion_dataset.csv", index=False)
    print(f"Generated {len(paired_df)} paired samples")
    print(paired_df["true_final_label"].value_counts())
