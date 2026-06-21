import os
import warnings
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, f1_score, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

import category_encoders as ce
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
MODELS_DIR = os.path.join("data", "processed", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DURATION_MODEL_PATH = os.path.join(MODELS_DIR, "duration_rf_model.pkl")
CLOSURE_MODEL_PATH = os.path.join(MODELS_DIR, "closure_rf_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
ENCODER_PATH = os.path.join(MODELS_DIR, "advanced_encoder.pkl")
KMEANS_PATH = os.path.join(MODELS_DIR, "advanced_kmeans.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "advanced_tfidf.pkl")
SVD_PATH = os.path.join(MODELS_DIR, "advanced_svd.pkl")

CONFIG = {
    "data_path": "data/processed/cleaned_events.csv",
    "datetime_col": "start_datetime",
    "lat_col": "latitude",
    "lon_col": "longitude",
    "text_cols": ["description", "comment"],
    "categorical_cols": ["corridor", "event_cause", "event_type", "priority", "corridor_cause", "corridor_priority", "cause_priority"],
    "numeric_cols": ["hour_of_day", "day_of_week", "is_weekend"],
    "regression_target": "duration_minutes",
    "classification_target": "requires_road_closure",
    "n_geo_clusters": 25,
    "tfidf_max_features": 100,
    "svd_components": 10,
    "optuna_trials_regression": 50,
    "optuna_trials_classification": 50,
}

def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data_path"])
    df[cfg["datetime_col"]] = pd.to_datetime(df[cfg["datetime_col"]])
    df["priority"] = df["priority"].fillna("Unknown")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["event_cause"] = df["event_cause"].fillna("unknown")
    df["event_type"] = df["event_type"].fillna("unplanned")
    df["description"] = df["description"].fillna("")
    df["comment"] = df["comment"].fillna("")
    df["latitude"] = df["latitude"].fillna(0.0)
    df["longitude"] = df["longitude"].fillna(0.0)
    df["requires_road_closure"] = df["requires_road_closure"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df

def add_cyclical_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def add_geo_clusters(df: pd.DataFrame, cfg: dict, is_train=True, km=None) -> (pd.DataFrame, KMeans):
    df = df.copy()
    coords = df[[cfg["lat_col"], cfg["lon_col"]]].values
    if is_train:
        km = KMeans(n_clusters=cfg["n_geo_clusters"], random_state=RANDOM_STATE, n_init=10)
        df["geo_cluster"] = km.fit_predict(coords).astype(str)
    else:
        df["geo_cluster"] = km.predict(coords).astype(str)
    return df, km

def add_text_features(df: pd.DataFrame, cfg: dict, is_train=True, vectorizer=None, svd=None):
    df = df.copy()
    combined_text = df[cfg["text_cols"]].agg(" ".join, axis=1)

    if is_train:
        vectorizer = SentenceTransformer("all-MiniLM-L6-v2")
        svd_matrix = vectorizer.encode(combined_text.tolist(), show_progress_bar=False)
        svd = None
    else:
        svd_matrix = vectorizer.encode(combined_text.tolist(), show_progress_bar=False)
        
    svd_cols = [f"svd_{i}" for i in range(svd_matrix.shape[1])]
    svd_df = pd.DataFrame(svd_matrix, columns=svd_cols, index=df.index)

    severity_terms = [
        "fatal", "waterlog", "water log", "flooded", "pileup", "closure", 
        "breakdown", "vip", "protest", "rally", "tree fall", "fallen tree", 
        "bwssb", "bescom", "bbmp", "bda", "white topping", "asphalting", 
        "accident", "lorry", "bus", "crane", "tow", "pothole"
    ]
    for term in severity_terms:
        col = f"kw_{term.replace('-', '_')}"
        df[col] = combined_text.str.contains(term, case=False, regex=False).astype(int)

    return pd.concat([df, svd_df], axis=1), vectorizer, svd

def engineer_features(df: pd.DataFrame, cfg: dict, is_train=True, km=None, vectorizer=None, svd=None):
    df = add_cyclical_features(df, cfg)
    df, km = add_geo_clusters(df, cfg, is_train, km)
    df, vectorizer, svd = add_text_features(df, cfg, is_train, vectorizer, svd)
    df["rainfall_mm"] = 0.0
    df["corridor_cause"] = df["corridor"].astype(str) + "_" + df["event_cause"].astype(str)
    df["corridor_priority"] = df["corridor"].astype(str) + "_" + df["priority"].astype(str)
    df["cause_priority"] = df["event_cause"].astype(str) + "_" + df["priority"].astype(str)
    return df, km, vectorizer, svd

def build_feature_lists(cfg: dict, df: pd.DataFrame):
    categorical_cols = cfg["categorical_cols"] + ["geo_cluster"]
    numeric_cols = (
        ["is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "rainfall_mm"]
        + [c for c in df.columns if c.startswith("svd_") or c.startswith("kw_")]
    )
    return categorical_cols, numeric_cols

def encode_features(df: pd.DataFrame, categorical_cols, numeric_cols, is_train=True, encoder=None, y=None):
    if is_train:
        encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=10)
        cat_encoded = encoder.fit_transform(df[categorical_cols], y)
    else:
        cat_encoded = encoder.transform(df[categorical_cols])

    X = np.hstack([cat_encoded.values, df[numeric_cols].values])
    cat_feature_indices = list(range(len(categorical_cols)))
    feature_names = categorical_cols + numeric_cols
    return X, cat_feature_indices, feature_names, encoder

class CloneableCatBoostRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, iterations=300, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
    def fit(self, X, y):
        self.model_ = CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, random_state=self.random_state, verbose=self.verbose)
        self.model_.fit(X, y)
        return self
    def predict(self, X):
        return self.model_.predict(X)
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

class CloneableCatBoostClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, iterations=300, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=False, scale_pos_weight=1.0):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.scale_pos_weight = scale_pos_weight
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, random_state=self.random_state, verbose=self.verbose, scale_pos_weight=self.scale_pos_weight)
        self.model_.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    def predict(self, X):
        return self.model_.predict(X)
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

def suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 9, 11]),
        "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 1.0]),
    }

def tune_xgb_regressor(X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = suggest_xgb_params(trial)
        base_model = XGBRegressor(**params, random_state=RANDOM_STATE, tree_method="hist", objective="reg:squarederror")
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return r2_score(y_val, preds)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Optuna/regression] best R^2={study.best_value:.4f}, params={study.best_params}")
    return study.best_params

def tune_xgb_classifier(X_train, y_train, n_trials, scale_pos_weight):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    def objective(trial):
        params = suggest_xgb_params(trial)
        model = XGBClassifier(**params, random_state=RANDOM_STATE, tree_method="hist", objective="binary:logistic", scale_pos_weight=scale_pos_weight)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[Optuna/classification] best F1={study.best_value:.4f}, params={study.best_params}")
    return study.best_params

def train_stacking_regressor(X_train, y_train, best_xgb_params):
    xgb_reg = XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, tree_method="hist", objective="reg:squarederror")
    lgbm_reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1)
    cat_reg = CloneableCatBoostRegressor(iterations=300, learning_rate=0.05, depth=6)

    stack = StackingRegressor(
        estimators=[("xgb", xgb_reg), ("lgbm", lgbm_reg), ("catboost", cat_reg)],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1, passthrough=False
    )
    transformed_stack = TransformedTargetRegressor(regressor=stack, func=np.log1p, inverse_func=np.expm1)
    transformed_stack.fit(X_train, y_train)
    return transformed_stack

def train_voting_classifier(X_train, y_train, best_xgb_params, scale_pos_weight):
    xgb_clf = XGBClassifier(**best_xgb_params, random_state=RANDOM_STATE, tree_method="hist", objective="binary:logistic", scale_pos_weight=scale_pos_weight)
    lgbm_clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1, scale_pos_weight=scale_pos_weight)
    cat_clf = CloneableCatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, scale_pos_weight=scale_pos_weight)
    rf_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE)

    vote = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgbm", lgbm_clf), ("catboost", cat_clf), ("rf", rf_clf)],
        voting="soft",
        n_jobs=-1
    )
    vote.fit(X_train, y_train)
    return vote

def main():
    cfg = CONFIG
    df = load_data(cfg)
    
    # 1. Feature Engineering (LSA replaces raw TF-IDF)
    df, km, vectorizer, svd = engineer_features(df, cfg, is_train=True)
    categorical_cols, numeric_cols = build_feature_lists(cfg, df)

    df["severe_impact"] = ((df[cfg["classification_target"]] == 1) | (df["duration_minutes"] >= 60)).astype(int)
    y_clf_all = df["severe_impact"].values

    # 2. Target Encoding (Using Classification Target)
    X, cat_idx, feature_names, encoder = encode_features(df, categorical_cols, numeric_cols, is_train=True, y=y_clf_all)
    
    # Save preprocessing artifacts
    joblib.dump(km, KMEANS_PATH)
    joblib.dump(vectorizer, TFIDF_PATH)
    joblib.dump(svd, SVD_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    
    # Ensure has_duration == True for regression
    dur_mask = df["has_duration"] == True
    X_reg = X[dur_mask]
    y_reg = df[dur_mask][cfg["regression_target"]].values
    
    y_clf = y_clf_all

    # --- REGRESSION ---
    print("Training Stacking Regressors (Planned vs Unplanned)...")
    df_dur = df[dur_mask]
    is_planned = (df_dur["event_type"] == "planned").values
    is_unplanned = (df_dur["event_type"] == "unplanned").values
    
    # Train Planned Model
    print("--- Planned Events Model ---")
    if is_planned.sum() > 0:
        X_reg_p, y_reg_p = X_reg[is_planned], y_reg[is_planned]
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_reg_p, y_reg_p, test_size=0.2, random_state=RANDOM_STATE)
        X_tr_p, X_val_p, y_tr_p, y_val_p = train_test_split(X_train_p, y_train_p, test_size=0.2, random_state=RANDOM_STATE)
        
        best_xgb_reg_params_p = tune_xgb_regressor(X_tr_p, y_tr_p, X_val_p, y_val_p, n_trials=cfg["optuna_trials_regression"])
        stack_reg_p = train_stacking_regressor(X_train_p, y_train_p, best_xgb_reg_params_p)
        
        test_preds_p = stack_reg_p.predict(X_test_p)
        test_r2_p = r2_score(y_test_p, test_preds_p)
        test_mae_p = mean_absolute_error(y_test_p, test_preds_p)
        print(f"Final StackingRegressor PLANNED test R^2: {test_r2_p:.4f}, MAE: {test_mae_p:.2f}")
        joblib.dump(stack_reg_p, DURATION_MODEL_PATH.replace("rf_model.pkl", "planned_rf_model.pkl"))
    else:
        test_preds_p, y_test_p = np.array([]), np.array([])
        test_mae_p = 0.0
        
    # Train Unplanned Model
    print("--- Unplanned Events Model ---")
    if is_unplanned.sum() > 0:
        X_reg_u, y_reg_u = X_reg[is_unplanned], y_reg[is_unplanned]
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_reg_u, y_reg_u, test_size=0.2, random_state=RANDOM_STATE)
        X_tr_u, X_val_u, y_tr_u, y_val_u = train_test_split(X_train_u, y_train_u, test_size=0.2, random_state=RANDOM_STATE)
        
        best_xgb_reg_params_u = tune_xgb_regressor(X_tr_u, y_tr_u, X_val_u, y_val_u, n_trials=cfg["optuna_trials_regression"])
        stack_reg_u = train_stacking_regressor(X_train_u, y_train_u, best_xgb_reg_params_u)
        
        test_preds_u = stack_reg_u.predict(X_test_u)
        test_r2_u = r2_score(y_test_u, test_preds_u)
        test_mae_u = mean_absolute_error(y_test_u, test_preds_u)
        print(f"Final StackingRegressor UNPLANNED test R^2: {test_r2_u:.4f}, MAE: {test_mae_u:.2f}")
        joblib.dump(stack_reg_u, DURATION_MODEL_PATH.replace("rf_model.pkl", "unplanned_rf_model.pkl"))
    else:
        test_preds_u, y_test_u = np.array([]), np.array([])
        test_mae_u = 0.0
        
    # Combined Metrics
    y_test_all = np.concatenate([y_test_p, y_test_u])
    test_preds_all = np.concatenate([test_preds_p, test_preds_u])
    if len(y_test_all) > 0:
        test_r2 = r2_score(y_test_all, test_preds_all)
        test_mae = mean_absolute_error(y_test_all, test_preds_all)
    else:
        test_r2, test_mae = 0.0, 0.0
    print(f"Combined Regression test R^2: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # --- CLASSIFICATION ---
    print("Training Super Voting Classifier with Target Encoding and SentenceTransformers...")
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)
    
    print("Applying SMOTE to balance classification classes...")
    smote = SMOTE(random_state=RANDOM_STATE)
    Xc_train_resampled, yc_train_resampled = smote.fit_resample(Xc_train, yc_train)
    
    spw = 1.0 # Since classes are perfectly balanced now

    best_xgb_clf_params = tune_xgb_classifier(Xc_train_resampled, yc_train_resampled, n_trials=cfg["optuna_trials_classification"], scale_pos_weight=spw)

    clf = train_voting_classifier(Xc_train_resampled, yc_train_resampled, best_xgb_clf_params, scale_pos_weight=spw)
    
    # Out-Of-Fold (OOF) threshold tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(clf, Xc_train_resampled, yc_train_resampled, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (oof_probs >= thresh).astype(int)
        f = f1_score(yc_train_resampled, preds)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
            
    print(f"Optimal Decision Threshold (via OOF): {best_thresh:.2f}")
    
    clf_probs = clf.predict_proba(Xc_test)[:, 1]
    clf_preds = (clf_probs >= best_thresh).astype(int)
    test_f1 = f1_score(yc_test, clf_preds)
    test_acc = accuracy_score(yc_test, clf_preds)
    test_prec = precision_score(yc_test, clf_preds, zero_division=0)
    test_rec = recall_score(yc_test, clf_preds, zero_division=0)
    
    print(f"Final VotingClassifier test F1: {test_f1:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    joblib.dump(clf, CLOSURE_MODEL_PATH)
    
    metrics = {
        "duration_mae_minutes": round(test_mae, 2),
        "duration_r2": round(test_r2, 4),
        "closure_accuracy": round(test_acc, 4),
        "closure_precision": round(test_prec, 4),
        "closure_recall": round(test_rec, 4),
        "closure_f1": round(test_f1, 4),
        "closure_optimal_threshold": round(best_thresh, 4),
        "training_samples": len(df)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("All models and artifacts saved!")

if __name__ == "__main__":
    main()
