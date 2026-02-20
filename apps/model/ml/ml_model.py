import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class CropRecommender:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Always use absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "Crop_recommendation.csv")
        df = pd.read_csv(csv_path)
        X = df.drop("label", axis=1)
        y = df["label"]
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_train, _, y_train, _ = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.feature_names = X.columns

    def recommend_crop_with_alternatives(self, input_data):
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_df)
        probs = self.model.predict_proba(input_scaled)[0]
        best_idx = np.argmax(probs)
        best_crop = self.encoder.inverse_transform([best_idx])[0]
        best_conf = probs[best_idx]
        variations = []
        for _ in range(100):
            noise = np.random.normal(0, 0.05, size=len(input_data))
            new_sample = np.array(input_data) * (1 + noise)
            variations.append(new_sample)
        variations_df = pd.DataFrame(variations, columns=self.feature_names)
        variations_scaled = self.scaler.transform(variations_df)
        preds = self.model.predict(variations_scaled)
        unique, counts = np.unique(preds, return_counts=True)
        crop_freq = dict(zip(unique, counts))
        crop_freq.pop(best_idx, None)
        sorted_alts = sorted(crop_freq.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        total = sum(counts)
        for idx, count in sorted_alts[:3]:
            crop_name = self.encoder.inverse_transform([idx])[0]
            confidence = count / total
            alternatives.append((crop_name, confidence))
        return best_crop, best_conf, alternatives