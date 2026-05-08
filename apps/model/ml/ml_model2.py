import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
	from tensorflow.keras.utils import to_categorical
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Dropout, Input
	from tensorflow.keras.callbacks import EarlyStopping
	_HAS_KERAS = True
except Exception:
	_HAS_KERAS = False

try:
	import shap
	_HAS_SHAP = True
except Exception:
	_HAS_SHAP = False


# Paths
BASE_DIR = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(BASE_DIR, "Crop_recommendation.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_dataset(csv_path: Optional[str] = None):
	path = csv_path or DEFAULT_CSV
	df = pd.read_csv(path)
	X = df.drop("label", axis=1)
	y = df["label"]
	return X, y


def preprocess(X: pd.DataFrame, scaler: Optional[MinMaxScaler] = None):
	if scaler is None:
		scaler = MinMaxScaler()
		X_scaled = scaler.fit_transform(X)
	else:
		X_scaled = scaler.transform(X)
	return X_scaled, scaler


def train_and_save_models(csv_path: Optional[str] = None, save=True):
	X, y = load_dataset(csv_path)

	le = LabelEncoder()
	y_encoded = le.fit_transform(y)

	X_scaled, scaler = preprocess(X)

	X_train, X_test, y_train, y_test = train_test_split(
		X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
	)

	smote = SMOTE(random_state=42)
	X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

	models = {
		"Naive Bayes": GaussianNB(),
		"Decision Tree": DecisionTreeClassifier(random_state=42),
		"Logistic Regression": LogisticRegression(max_iter=500),
		"SVM": SVC(probability=True),
		"Random Forest": RandomForestClassifier(random_state=42),
		"Gradient Boosting": GradientBoostingClassifier(random_state=42),
		"XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
	}

	results = []
	fitted = {}

	for name, model in models.items():
		model.fit(X_train_res, y_train_res)
		cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
		y_pred = model.predict(X_test)

		acc = accuracy_score(y_test, y_pred)

		results.append({
			"model": name,
			"accuracy": acc,
			"cv_mean": float(cv_scores.mean()),
			"cv_std": float(cv_scores.std())
		})
		fitted[name] = model

	# Grid search for Random Forest
	param_grid = {
		'n_estimators': [200, 300],
		'max_depth': [None, 20, 30],
		'min_samples_split': [2, 5]
	}
	rf = RandomForestClassifier(random_state=42)
	grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
	grid.fit(X_train_res, y_train_res)
	best_rf = grid.best_estimator_
	fitted['Random Forest (Tuned)'] = best_rf

	# Ensemble (soft voting)
	ensemble = VotingClassifier(
		estimators=[
			('rf', best_rf),
			('svm', SVC(probability=True)),
			('xgb', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
		],
		voting='soft'
	)
	ensemble.fit(X_train_res, y_train_res)
	fitted['Ensemble'] = ensemble

	# Optionally train a small neural network if Keras is available
	nn_path = None
	if _HAS_KERAS:
		y_cat = to_categorical(y_encoded)
		X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
			X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
		)

		model_nn = Sequential([
			Input(shape=(X_train_nn.shape[1],)),
			Dense(128, activation='relu'),
			Dropout(0.4),
			Dense(64, activation='relu'),
			Dropout(0.3),
			Dense(32, activation='relu'),
			Dense(y_cat.shape[1], activation='softmax')
		])
		model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
		history = model_nn.fit(
			X_train_nn, y_train_nn,
			epochs=100,
			batch_size=32,
			validation_split=0.2,
			callbacks=[early],
			verbose=0
		)
		nn_path = os.path.join(ARTIFACT_DIR, 'model_nn.h5')
		model_nn.save(nn_path)
		fitted['NeuralNet'] = model_nn

	# Save artifacts
	if save:
		joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.joblib'))
		joblib.dump(le, os.path.join(ARTIFACT_DIR, 'label_encoder.joblib'))
		# Save fitted traditional models
		for name, m in fitted.items():
			# skip Keras model (saved separately)
			if name == 'NeuralNet':
				continue
			fname = os.path.join(ARTIFACT_DIR, f"model_{name.replace(' ', '_')}.joblib")
			try:
				joblib.dump(m, fname)
			except Exception:
				# some models (like xgboost sklearn wrapper) may not pickle in older configs
				pass

	return {
		'results': results,
		'fitted': fitted,
		'scaler': scaler,
		'label_encoder': le,
		'nn_path': nn_path
	}


def _load_artifact(name: str):
	path = os.path.join(ARTIFACT_DIR, name)
	if os.path.exists(path):
		return joblib.load(path)
	return None


def predict_crop(features: Union[pd.DataFrame, list, np.ndarray], model_name: str = 'Ensemble') -> dict:
	"""Predict crop label and code from a single sample or batch.

	features: 1D/list or 2D array or DataFrame
	model_name: name of saved model file prefix (default 'Ensemble' matches saved artifact)
	Returns: dict with keys `code`, `label`, `proba` (probabilities)
	"""
	# load scaler and encoder
	scaler = _load_artifact('scaler.joblib')
	le = _load_artifact('label_encoder.joblib')
	feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
	
	model = None
	model_file = os.path.join(ARTIFACT_DIR, f"model_{model_name.replace(' ', '_')}.joblib")
	if os.path.exists(model_file):
		model = joblib.load(model_file)

	if model is None:
		# fallback: try ensemble or Random Forest
		for candidate in ['model_Ensemble.joblib', 'model_Random_Forest.joblib', 'model_Random_Forest_(Tuned).joblib']:
			p = os.path.join(ARTIFACT_DIR, candidate)
			if os.path.exists(p):
				model = joblib.load(p)
				break

	if model is None or scaler is None or le is None:
		raise RuntimeError('Required artifacts not found. Run training via train_and_save_models() first.')

	# normalize input
	if isinstance(features, pd.DataFrame):
		X = features.values
	else:
		X = np.array(features)
	if X.ndim == 1:
		X = X.reshape(1, -1)
	# Convert to DataFrame with feature names to avoid sklearn warning
	X = pd.DataFrame(X, columns=feature_names)

	X_scaled = scaler.transform(X)

	if hasattr(model, 'predict_proba'):
		proba = model.predict_proba(X_scaled)
	else:
		# last resort: use decision function to approximate
		try:
			dec = model.decision_function(X_scaled)
			# softmax
			exp = np.exp(dec - np.max(dec, axis=1, keepdims=True))
			proba = exp / exp.sum(axis=1, keepdims=True)
		except Exception:
			proba = None

	pred = model.predict(X_scaled)
	code = int(pred[0])
	label = le.inverse_transform([code])[0]

	# Build top-K alternatives from probabilities to be comparable with best probability
	alternatives = []
	if proba is not None:
		probs_row = np.array(proba)[0]
		# get top 4 indices (including best) then exclude best when presenting alternatives
		top_idx = probs_row.argsort()[::-1]
		# prepare alternatives as list of (label, confidence) excluding predicted class
		for idx in top_idx[1:4]:
			alt_label = le.inverse_transform([int(idx)])[0]
			alt_conf = float(probs_row[int(idx)])
			alternatives.append({'crop': alt_label, 'confidence': alt_conf})

	# Compute SHAP values for feature importance (if available and model is tree-based)
	shap_values_list = []
	if _HAS_SHAP and hasattr(model, 'predict'):
		try:
			# Try to use TreeExplainer for tree-based models (RF, XGB ensemble member)
			if hasattr(model, 'estimators_') or 'RandomForest' in str(type(model)):
				explainer = shap.TreeExplainer(model)
				shap_vals = explainer.shap_values(X_scaled)
				# For multiclass, shap_vals is a list of arrays; take the predicted class
				if isinstance(shap_vals, list):
					shap_vals = shap_vals[code]
				elif shap_vals.ndim == 3:
					shap_vals = shap_vals[0, :, code]
				else:
					shap_vals = shap_vals[0, :]
				
				# Build feature importance list sorted by abs(shap_value)
				feature_names_list = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
				shap_importance = []
				for fname, sval in zip(feature_names_list, shap_vals):
					shap_importance.append({'feature': fname, 'value': float(sval)})
				# Sort by abs value descending
				shap_importance.sort(key=lambda x: abs(x['value']), reverse=True)
				shap_values_list = shap_importance
		except Exception:
			pass

	return {
		'code': code,
		'label': label,
		'proba': proba.tolist() if proba is not None else None,
		'alternatives': alternatives,
		'shap_values': shap_values_list,
	}


if __name__ == '__main__':
	# When run directly, train models and save artifacts.
	print('Training models (this may take a while)...')
	out = train_and_save_models()
	print('Training finished. Models and artifacts saved to', ARTIFACT_DIR)

