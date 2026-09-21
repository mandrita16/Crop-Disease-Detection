import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


class FusionModel:
    """
    Meta-learner that takes [image_model_probs, tabular_model_probs]
    concatenated together and learns how to weigh them per-class,
    rather than a hand-picked average. Also flags disagreement cases
    for manual review.
    """

    def __init__(self):
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit(self, image_probs, tabular_probs, y_true):
        X_meta = np.hstack([image_probs, tabular_probs])
        self.meta_model.fit(X_meta, y_true)

    def predict(self, image_probs, tabular_probs, disagreement_threshold=0.3):
        X_meta = np.hstack([image_probs, tabular_probs])
        final_pred = self.meta_model.predict(X_meta)
        final_proba = self.meta_model.predict_proba(X_meta)

        # Disagreement flag: image and tabular branches predict different top class
        image_top = np.argmax(image_probs, axis=1)
        tabular_top = np.argmax(tabular_probs, axis=1)
        disagreement = image_top != tabular_top

        results = []
        for i in range(len(final_pred)):
            results.append({
                "final_prediction": int(final_pred[i]),
                "confidence": float(np.max(final_proba[i])),
                "needs_review": bool(disagreement[i]),
            })
        return results

    def save(self, path="../models/fusion_meta_model.pkl"):
        joblib.dump(self.meta_model, path)

    def load(self, path="../models/fusion_meta_model.pkl"):
        self.meta_model = joblib.load(path)
