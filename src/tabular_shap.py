import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from tabular_data import load_and_prepare


def explain():
    X_train, X_test, y_train, y_test, features, df = load_and_prepare()

    model = xgb.XGBClassifier()
    model.load_model("../models/tabular_xgb_model.json")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig("../static/shap_summary.png")
    print("SHAP summary saved to static/shap_summary.png")

    return explainer, shap_values


def explain_single(sample_dict):
    """
    sample_dict: e.g. {"N": 40, "P": 25, "K": 20, "temperature": 32,
                        "humidity": 45, "ph": 5.2, "rainfall": 80}
    Returns per-feature contribution for one prediction, plus the full
    probability vector (needed by the fusion model).
    """
    scaler = joblib.load("../models/tabular_scaler.pkl")
    le = joblib.load("../models/tabular_label_encoder.pkl")
    model = xgb.XGBClassifier()
    model.load_model("../models/tabular_xgb_model.json")

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    row = pd.DataFrame([sample_dict])[features]
    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]
    pred_label = le.inverse_transform([pred])[0]
    probs = model.predict_proba(row_scaled)[0]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row_scaled)

    contributions = dict(zip(features, shap_vals[pred][0] if isinstance(shap_vals, list) else shap_vals[0]))

    return {
        "predicted_label": pred_label,
        "confidence": float(max(probs)),
        "all_probs": probs.tolist(),
        "class_names": list(le.classes_),
        "feature_contributions": {k: round(float(v), 4) for k, v in contributions.items()},
    }


if __name__ == "__main__":
    explain()
    result = explain_single({"N": 40, "P": 25, "K": 20, "temperature": 32,
                              "humidity": 45, "ph": 5.2, "rainfall": 80})
    print(result)
