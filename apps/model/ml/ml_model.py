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
        best_idx = int(np.argmax(probs))
        best_crop = self.encoder.inverse_transform([best_idx])[0]
        best_conf = float(probs[best_idx])

        # Build alternatives from top probabilities (excluding best)
        alternatives = []
        top_idx = probs.argsort()[::-1]
        for idx in top_idx[1:4]:
            alt_name = self.encoder.inverse_transform([int(idx)])[0]
            alt_conf = float(probs[int(idx)])
            alternatives.append((alt_name, alt_conf))

        return best_crop, best_conf, alternatives