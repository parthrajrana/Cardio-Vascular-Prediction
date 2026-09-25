import time
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory where trained models and results are cached
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# Feature column order expected by the models
FEATURE_COLUMNS = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

RANDOM_STATE = 42

# Global in-memory storage for trained models and Task 5 evaluation results
TRAINED_MODELS = {}
MODEL_RESULTS = []
MODEL_RESULTS_DICT = {}
BEST_MODEL_NAME = ""
TUNING_RESULTS = {}
IS_TRAINED = False


def load_and_preprocess_data(dataset_path: str = "cardio_train.csv"):
    """
    Loads cardio_train.csv, drops non-predictive columns,
    and converts age from days to years.
    Also removes physiologically impossible outlier rows that corrupt
    the StandardScaler and cause misclassification of healthy inputs.
    """
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["cardio", "id"]).copy()
    y = df["cardio"]

    # Convert age from days to years for consistency with clinical inputs
    X["age"] = X["age"] / 365.25

    # ── Clinical Outlier Removal ──────────────────────────────────────
    # The raw dataset contains physiologically impossible values (e.g.,
    # ap_hi=11000, ap_lo=-70) that skew the StandardScaler and cause
    # healthy patients to be misclassified as high-risk.
    mask = (
        (X["ap_hi"] >= 60)   & (X["ap_hi"] <= 250) &   # Systolic BP: clinically valid range
        (X["ap_lo"] >= 40)   & (X["ap_lo"] <= 150) &   # Diastolic BP: clinically valid range
        (X["ap_hi"] > X["ap_lo"])                   &   # Systolic must exceed diastolic
        (X["height"] >= 100) & (X["height"] <= 220) &   # Height in cm
        (X["weight"] >= 30)  & (X["weight"] <= 200) &   # Weight in kg
        (X["age"] >= 1)      & (X["age"] <= 100)         # Age in years
    )
    X = X[mask]
    y = y[mask]
    print(f"[ML] Data cleaned: kept {mask.sum():,} / {len(mask):,} rows after outlier removal")

    return X, y



def get_base_models():
    """
    Defines the classification pipelines required by Task 5.
    Each model is paired with a StandardScaler in a Pipeline.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=60,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", AdaBoostClassifier(
                n_estimators=50,
                random_state=RANDOM_STATE
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ])
    }


def determine_fit_status(train_score: float, test_score: float) -> str:
    """
    Evaluates overfitting / underfitting according to Task 5 rules:
    - Train score much higher than test score (> 0.05 gap) -> Possible overfitting
    - Both scores low (< 0.65) -> Possible underfitting
    - Train and test scores close -> Good fit
    """
    gap = train_score - test_score
    if gap > 0.05:
        return "Possible overfitting"
    elif train_score < 0.65 and test_score < 0.65:
        return "Possible underfitting"
    else:
        return "Good fit"


def _cache_path(filename: str) -> str:
    """Returns the full path to a cache file."""
    return os.path.join(CACHE_DIR, filename)


def _load_from_cache() -> bool:
    """
    Attempts to load all trained models and results from the model_cache directory.
    Returns True if cache was loaded successfully, False otherwise.
    """
    global TRAINED_MODELS, MODEL_RESULTS, MODEL_RESULTS_DICT, BEST_MODEL_NAME, TUNING_RESULTS, IS_TRAINED

    results_path = _cache_path("results.json")
    if not os.path.exists(results_path):
        return False

    try:
        print("[ML] Cache found! Loading models from disk...")
        # Load results metadata
        with open(results_path, "r") as f:
            cache = json.load(f)

        MODEL_RESULTS = cache["model_results"]
        MODEL_RESULTS_DICT = cache["model_results_dict"]
        BEST_MODEL_NAME = cache["best_model_name"]
        TUNING_RESULTS = cache["tuning_results"]

        # Load each trained model pipeline
        for name in list(MODEL_RESULTS_DICT.keys()) + [f"{cache['best_model_name']} (Tuned)"]:
            pkl_file = _cache_path(f"{name.replace(' ', '_')}.pkl")
            if os.path.exists(pkl_file):
                TRAINED_MODELS[name] = joblib.load(pkl_file)
                print(f"[ML] Loaded: {name}")

        IS_TRAINED = True
        print("[ML] All models loaded from cache successfully!")
        return True
    except Exception as e:
        print(f"[ML] Cache load failed ({e}), will retrain...")
        return False


def _save_to_cache():
    """
    Saves all trained models and results metadata to the model_cache directory.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Save each model pipeline as a .pkl file
    for name, model in TRAINED_MODELS.items():
        pkl_file = _cache_path(f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, pkl_file)
        print(f"[ML] Saved: {name} -> {pkl_file}")

    # Save results metadata as JSON
    cache = {
        "model_results": MODEL_RESULTS,
        "model_results_dict": MODEL_RESULTS_DICT,
        "best_model_name": BEST_MODEL_NAME,
        "tuning_results": TUNING_RESULTS
    }
    with open(_cache_path("results.json"), "w") as f:
        json.dump(cache, f, indent=2)

    print(f"[ML] All models and results saved to: {CACHE_DIR}")


def init_ml_models(dataset_path: str = "cardio_train.csv"):
    """
    Initializes all ML models.
    - First run: trains all models, runs 5-fold CV, tunes the best model, then saves to disk.
    - Subsequent runs: loads instantly from disk cache (model_cache/ folder).
    To force retraining, delete the model_cache/ folder.
    """
    global TRAINED_MODELS, MODEL_RESULTS, MODEL_RESULTS_DICT, BEST_MODEL_NAME, TUNING_RESULTS, IS_TRAINED

    # Try loading from cache first
    if _load_from_cache():
        return

    print("[ML] No cache found. Starting full training (this only happens once)...")
    start_time = time.time()

    # 1. Load and split dataset
    X, y = load_and_preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 2. 5-Fold Stratified Cross-Validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = get_base_models()
    results = []
    results_dict = {}

    # 3. Train and evaluate all 4 models
    for name, model in models.items():
        m_start = time.time()
        # Fit on training data
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_acc = float(accuracy_score(y_train, train_pred))
        test_acc = float(accuracy_score(y_test, test_pred))
        prec = float(precision_score(y_test, test_pred, zero_division=0))
        rec = float(recall_score(y_test, test_pred, zero_division=0))
        f1 = float(f1_score(y_test, test_pred, zero_division=0))

        # 5-fold Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        fit_status = determine_fit_status(train_acc, test_acc)

        metric_entry = {
            "name": name,
            "accuracy": round(test_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "train_score": round(train_acc, 4),
            "test_score": round(test_acc, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "fit_status": fit_status
        }

        results.append(metric_entry)
        results_dict[name] = metric_entry
        TRAINED_MODELS[name] = model

        print(f"[ML] {name:20s} | Test Acc: {test_acc:.4f} | CV Mean: {cv_mean:.4f} (std: {cv_std:.4f}) | Fit: {fit_status} ({time.time()-m_start:.2f}s)")

    # Sort results by highest CV Mean, then lowest CV Std
    results.sort(key=lambda r: (-r["cv_mean"], r["cv_std"]))
    MODEL_RESULTS = results
    MODEL_RESULTS_DICT = results_dict

    # 4. Select model for tuning based on CV Mean
    BEST_MODEL_NAME = results[0]["name"]
    best_base_model = TRAINED_MODELS[BEST_MODEL_NAME]
    print(f"[ML] Selected Model for Tuning based on CV: {BEST_MODEL_NAME} (CV Mean: {results[0]['cv_mean']:.4f})")

    # 5. Hyperparameter Tuning using focused GridSearchCV
    tuning_grids = {
        "Gradient Boosting": {
            "model__n_estimators": [50, 75],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4]
        },
        "Random Forest": {
            "model__n_estimators": [50, 80],
            "model__max_depth": [8, 12],
            "model__min_samples_split": [2, 5]
        },
        "AdaBoost": {
            "model__n_estimators": [50, 80],
            "model__learning_rate": [0.05, 0.1, 0.2]
        },
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 5.0],
            "model__solver": ["lbfgs"]
        }
    }

    grid = tuning_grids.get(BEST_MODEL_NAME, {
        "model__n_estimators": [50, 75],
        "model__learning_rate": [0.05, 0.1]
    })

    tune_search = GridSearchCV(
        estimator=best_base_model,
        param_grid=grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    tune_search.fit(X_train, y_train)

    tuned_model = tune_search.best_estimator_
    tuned_test_pred = tuned_model.predict(X_test)
    after_tuning_score = float(accuracy_score(y_test, tuned_test_pred))
    before_tuning_score = results_dict[BEST_MODEL_NAME]["test_score"]
    diff = round(after_tuning_score - before_tuning_score, 4)

    # Format best parameters for clean display
    clean_params = {
        k.replace("model__", ""): v for k, v in tune_search.best_params_.items()
    }

    TUNING_RESULTS = {
        "model_name": BEST_MODEL_NAME,
        "best_params": clean_params,
        "before_tuning_score": round(before_tuning_score, 4),
        "after_tuning_score": round(after_tuning_score, 4),
        "improvement": diff,
        "improvement_percent": f"{diff * 100:+.2f}%",
        "best_cv_score": round(float(tune_search.best_score_), 4),
        "tuned_precision": round(float(precision_score(y_test, tuned_test_pred, zero_division=0)), 4),
        "tuned_recall": round(float(recall_score(y_test, tuned_test_pred, zero_division=0)), 4),
        "tuned_f1": round(float(f1_score(y_test, tuned_test_pred, zero_division=0)), 4)
    }

    # Store tuned model under both its specific key and update active best
    TRAINED_MODELS[f"{BEST_MODEL_NAME} (Tuned)"] = tuned_model

    IS_TRAINED = True
    print(f"[ML] Hyperparameter Tuning completed: Best Params: {clean_params}")
    print(f"[ML] Score Before: {before_tuning_score:.4f} | After: {after_tuning_score:.4f} | Diff: {diff:+.4f}")
    print(f"[ML] Full ML training completed in {time.time()-start_time:.2f}s")

    # Save models to disk so future restarts load instantly
    _save_to_cache()


def get_available_models():
    """Returns list of all available trained models."""
    if not IS_TRAINED:
        init_ml_models()
    return [name for name in ["Logistic Regression", "Random Forest", "AdaBoost", "Gradient Boosting"] if name in TRAINED_MODELS]


def get_model_results():
    """Returns the evaluation results of every model."""
    if not IS_TRAINED:
        init_ml_models()
    return {
        "models": MODEL_RESULTS,
        "best_model": BEST_MODEL_NAME,
        "cv_folds": 5,
        "tuning": TUNING_RESULTS
    }


def get_model_details(model_name: str):
    """Returns complete results and hyperparameter tuning info for a particular model."""
    if not IS_TRAINED:
        init_ml_models()

    if model_name not in MODEL_RESULTS_DICT:
        return None

    detail = dict(MODEL_RESULTS_DICT[model_name])
    if model_name == BEST_MODEL_NAME:
        detail["tuning"] = TUNING_RESULTS
    else:
        detail["tuning"] = None
    return detail


def predict_disease(model_name: str, features: dict):
    """
    Accepts model name and dictionary of 11 clinical features.
    Returns disease prediction, confidence probability, and message.
    """
    if not IS_TRAINED:
        init_ml_models()

    if model_name not in TRAINED_MODELS:
        raise ValueError(f"Invalid model selected: '{model_name}'. Available: {get_available_models()}")

    model = TRAINED_MODELS[model_name]

    # Validate and organize features in exact order
    missing = [f for f in FEATURE_COLUMNS if f not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    row_data = {col: [float(features[col])] for col in FEATURE_COLUMNS}
    input_df = pd.DataFrame(row_data)

    pred_code = int(model.predict(input_df)[0])

    # Probability prediction if available
    probability = None
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        # Probability for positive class (disease presence)
        pos_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        probability = round(pos_prob, 4)
        # Confidence is the probability of the predicted class
        pred_prob = float(probs[pred_code])
        confidence = round(pred_prob * 100, 1)

    prediction_label = "Cardiovascular Disease Detected" if pred_code == 1 else "No Cardiovascular Disease Detected"

    return {
        "model": model_name,
        "prediction": prediction_label,
        "prediction_code": pred_code,
        "probability": probability,
        "confidence": confidence,
        "message": "Prediction successful"
    }
