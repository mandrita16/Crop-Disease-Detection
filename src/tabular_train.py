import xgboost as xgb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

from tabular_data import load_and_prepare

X_train, X_test, y_train, y_test, features, df = load_and_prepare()

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
joblib.dump(le, "../models/tabular_label_encoder.pkl")


def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "eval_metric": "mlogloss",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train_enc)
    preds = model.predict(X_test)
    return accuracy_score(y_test_enc, preds)


print("Tuning hyperparameters with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest params:", study.best_params)
print("Best accuracy:", study.best_value)

# Train final model with best params
best_params = study.best_params
best_params.update({
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "eval_metric": "mlogloss",
    "random_state": 42,
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train_enc)

preds = final_model.predict(X_test)
print("\n", classification_report(y_test_enc, preds, target_names=le.classes_))

final_model.save_model("../models/tabular_xgb_model.json")
with open("../models/tabular_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Model saved to models/tabular_xgb_model.json")
