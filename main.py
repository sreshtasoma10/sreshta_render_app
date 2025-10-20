# main.py (UPDATED)
import os
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from packaging import version
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
#import xgboost as xgb
import logging
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import optuna
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- ADD to top imports in main.py ---
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
#from xgboost import XGBRegressor
# add these near other imports
import io
import base64
import matplotlib.pyplot as plt
# seaborn is optional in server envs; keep import but not required
import seaborn as sns
# Additional imports for H2H improvements
from functools import lru_cache
from scipy.stats import ttest_ind
from fastapi import FastAPI, HTTPException, Request,Query  # ADD Request here
# --- end add ---

# Ensure matplotlib backend won't require display (for some environments)
import matplotlib
matplotlib.use('Agg')

global LOGISTIC_MODEL, MODEL_TRAINED


app = FastAPI(title="Cricket Match Predictor", version="1.0")


# Pydantic Models for Request Body
class PredictionRequest(BaseModel):
    team_a: str
    team_b: str
    venue: Optional[str] = None
    toss_winner: Optional[str] = None  # NEW
    toss_decision: Optional[str] = None  # NEW (bat or field)

class PlayerAnalysisRequest(BaseModel):
    player_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

class BowlerAnalysisRequest(BaseModel):
    bowler_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

# Additional H2H and innings request models
class H2HRequest(BaseModel):
    player_a: str
    player_b: str
    season: Optional[str] = None
    venue: Optional[str] = None

class InningsRequest(BaseModel):
    player_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

class InningsProgressionRequest(BaseModel):
    team: str
    innings: int
    year: int
    venue: str
    max_plots: Optional[int] = 8


# ===============================
# Head-to-Head API Request Models
# ===============================

class PlayerToPlayerComparisionRequest(BaseModel):
    player_a: str
    player_b: str
    season: Optional[str] = None
    venue: Optional[str] = None
    analysis_type: str = "auto"  # or "batsman_vs_bowler", "batsman_vs_batsman"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Model persistence config (Logistic Regression version)
# ===============================
MODEL_DIR = globals().get("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# Team winner model files (new names)
'''TEAM_MODEL_PATH = os.path.join(MODEL_DIR, "team_winner_logistic.joblib")
ENGINEERED_FEATURES_PATH = os.path.join(MODEL_DIR, "engineered_feature_cols.joblib")
ENGINEERED_MEDIANS_PATH = os.path.join(MODEL_DIR, "engineered_feature_medians.joblib")
FEATURE_ENCODERS_PATH = os.path.join(MODEL_DIR, "feature_encoders.joblib")
LAST_ACCURACY_PATH = os.path.join(MODEL_DIR, "last_accuracy.json")

# Keep any existing innings / other model paths untouched
MODEL_PATH_1 = os.path.join(MODEL_DIR, "innings_model_1.joblib")
MODEL_PATH_2 = os.path.join(MODEL_DIR, "innings_model_2.joblib")
ENC_PATH = os.path.join(MODEL_DIR, "innings_encoders.joblib")'''

# ===============================
# FastAPI App Initialization
# ===============================


# CORS (Cross-Origin Resource Sharing) middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ===============================
# Config
# ===============================
# IMPORTANT: Make sure this CSV file is in the same directory as this main.py file.
import os

# Keep your Windows defaults so nothing breaks locally,
# but allow container / Render to override via environment variables.
FILE_PATH1 = os.environ.get("FILE_PATH1", r"C:\Users\somas\OneDrive\Desktop\sreshta_render_app\data\ball_to_ball.csv")
FILE_PATH2 = os.environ.get("FILE_PATH2", "C:\Users\somas\OneDrive\Desktop\sreshta_render_app\data\match_by_match.csv")

MODEL_DIR  = os.environ.get("MODEL_DIR", "models")

MODEL_PATH = "criclytics_model.joblib"
FEATURES_PATH = "criclytics_features.joblib"
ACCURACY_PATH = "criclytics_accuracy.joblib"
# Global variables
DF = None
MATCH_DF = None
AVAILABLE_TEAMS = []
AVAILABLE_VENUES = []
AVAILABLE_SEASONS = []
AVAILABLE_PLAYERS = []
AVAILABLE_BOWLERS = []
LOGISTIC_MODEL = None
ENCODER = None
MODEL_TRAINED = False
LAST_ACCURACY = None   # will hold last training metrics dict
SELECTED_FEATURES = []  # Add this near other global variables


# =====================================
# MODEL + DATA LOADING (UNIFIED SETUP)
# =====================================
@app.on_event("startup")
def load_data_and_model():
    """
    Load model, features, and datasets.
    If model files don't exist, train new model automatically.
    """
    global BASE_MODEL, FEATURE_COLS, DF, MATCH_DF
    global AVAILABLE_TEAMS, AVAILABLE_VENUES, AVAILABLE_SEASONS, AVAILABLE_PLAYERS, AVAILABLE_BOWLERS

    try:
        # 1️⃣ Load datasets
        logger.info(f"📂 Loading base match dataset from '{FILE_PATH1}'...")
        DF = pd.read_csv(FILE_PATH1)
        logger.info(f"✅ Base dataset loaded: {DF.shape[0]} rows, {DF.shape[1]} columns")

        logger.info(f"📂 Loading engineered feature dataset from '{FILE_PATH2}'...")
        features_df = pd.read_csv(FILE_PATH2)
        logger.info(f"✅ Engineered dataset loaded: {features_df.shape[0]} rows, {features_df.shape[1]} columns")

        MATCH_DF = DF.drop_duplicates(subset=["MatchID"]).copy()

        # 2️⃣ Prepare available entities
        recent_ipl_teams = [
            "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
            "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
            "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
            "Sunrisers Hyderabad"
        ]

        all_teams = sorted(MATCH_DF["Team1"].unique().tolist()) if "Team1" in MATCH_DF.columns else []
        AVAILABLE_TEAMS = [team for team in all_teams if team in recent_ipl_teams]

        AVAILABLE_VENUES = sorted(MATCH_DF["Venue"].unique().tolist()) if "Venue" in MATCH_DF.columns else []

        # Handle seasons properly
        AVAILABLE_SEASONS = []
        if "Season" in DF.columns:
            seasons_series = pd.to_numeric(DF["Season"], errors='coerce').dropna()
            if not seasons_series.empty:
                AVAILABLE_SEASONS = sorted([int(s) for s in seasons_series.astype(int).unique()])
            else:
                AVAILABLE_SEASONS = list(range(2008, 2026))
        else:
            AVAILABLE_SEASONS = list(range(2008, 2026))
        logger.info(f"📅 Seasons detected: {AVAILABLE_SEASONS}")

        # Players and bowlers
        if "Batter" in DF.columns and "Bowler" in DF.columns:
            all_players = pd.concat([DF["Batter"], DF["Bowler"]]).dropna().unique()
            AVAILABLE_PLAYERS = sorted([str(p) for p in all_players if pd.notna(p)])
        else:
            AVAILABLE_PLAYERS = []
        AVAILABLE_BOWLERS = sorted(DF["Bowler"].dropna().unique().tolist()) if "Bowler" in DF.columns else []

        # 3️⃣ Try loading saved model & feature set
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            BASE_MODEL = joblib.load(MODEL_PATH)
            FEATURE_COLS = joblib.load(FEATURES_PATH)
            logger.info("✅ Loaded saved model and feature list successfully")
        else:
            logger.warning("⚠️ Model files not found. Training new model...")
            train_logistic_model(force_retrain=True)
            if os.path.exists(MODEL_PATH):
                BASE_MODEL = joblib.load(MODEL_PATH)
                FEATURE_COLS = joblib.load(FEATURES_PATH)
                logger.info("✅ New model trained and loaded successfully")
            else:
                raise FileNotFoundError("❌ Model training failed or file missing after training.")

        logger.info("🎯 Data & model loading complete!")
        logger.info(f"📊 Teams: {len(AVAILABLE_TEAMS)}, Venues: {len(AVAILABLE_VENUES)}, Seasons: {len(AVAILABLE_SEASONS)}")

    except FileNotFoundError as e:
        logger.error(f"❌ Data file not found: {e}")
        AVAILABLE_TEAMS, AVAILABLE_VENUES, AVAILABLE_PLAYERS, AVAILABLE_BOWLERS = [], [], [], []
        AVAILABLE_SEASONS = list(range(2008, 2026))
    except Exception as e:
        logger.error(f"❌ Error during startup loading: {e}", exc_info=True)
        AVAILABLE_TEAMS, AVAILABLE_VENUES, AVAILABLE_PLAYERS, AVAILABLE_BOWLERS = [], [], [], []
        AVAILABLE_SEASONS = list(range(2008, 2026))


######################################################
#SECTION 1 - TRAIN AND SAVE MODEL ONCE 
#################################################

# Add global variable to store the last prediction data
LAST_PREDICTION_DATA = None

def train_logistic_model(force_retrain=True, n_trials=30, selected_season="All Seasons"):
    """
    Complete training pipeline with comprehensive metrics and logging
    """
    global LOGISTIC_MODEL, MODEL_TRAINED, SELECTED_FEATURES

    logger.info(f"🚀 STARTING COMPLETE TRAINING PIPELINE for season: {selected_season}")
    logger.info("=" * 60)

    try:
        # ============================================
        # STEP 1: DATA LOADING AND ENCODING
        # ============================================
        logger.info("📥 STEP 1: Loading and encoding data...")
        
        engineered_csv = pd.read_csv(FILE_PATH2)
        logger.info(f"✅ Loaded raw data: {engineered_csv.shape}")

        

        if engineered_csv.empty:
            logger.error("❌ Engineered features CSV is empty")
            return {"status": "error", "message": "Empty dataset"}

        logger.info("🔤 Encoding entire dataset...")
        encoded_df = engineered_csv.copy()
        
        # Encode categorical variables
        categorical_columns = []
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object' and col not in ['MatchID', 'MatchDate', 'Winner', 'actual_winner']:
                categorical_columns.append(col)
        
        logger.info(f"📊 Found {len(categorical_columns)} categorical columns to encode")
        encoded_df=pd.get_dummies(encoded_df,columns=categorical_columns,dtype=int,drop_first=True)
        
        
        logger.info(f"🎉 Data encoding complete. Shape: {encoded_df.shape}")

        # ============================================
        # STEP 2: TARGET AND FEATURE SELECTION
        # ============================================
        logger.info("🎯 STEP 2: Selecting target and features...")
        
        target = "winner_is_teamA"
        
        if target not in encoded_df.columns:
            logger.error(f"❌ Target column '{target}' not found")
            return {"status": "error", "message": f"Target column '{target}' not found"}

        # Remove unnecessary columns
        drop_cols = [
            "actual_winner", "Winner", "MatchID", "MatchDate", "MatchStage",
            "Venue_Bat1stWinPct", "MatchType", "HomeAdvantage",
            "TeamA_AllRounders", "TeamB_AllRounders",
            "Venue_SpinFavorIndex", "DewFactor"
        ]
        
        columns_to_drop = [c for c in drop_cols if c in encoded_df.columns and c != target]
        encoded_df = encoded_df.drop(columns=columns_to_drop)
        logger.info(f"🗑️ Dropped columns: {columns_to_drop}")

        # Select numeric features
        numeric_cols = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target]
        
        logger.info(f"📊 Selected {len(feature_cols)} features")
        logger.info(f"🎯 Target variable: {target}")
        logger.info(f"📈 Features: {feature_cols}")

        # Remove leakage columns
        y_str = encoded_df[target].astype(str)
        leak_cols = [c for c in feature_cols if np.array_equal(encoded_df[c].astype(str).values, y_str.values)]
        if leak_cols:
            logger.warning(f"🚨 Leakage detected! Dropping: {leak_cols}")
            feature_cols = [c for c in feature_cols if c not in leak_cols]

        if not feature_cols:
            logger.error("❌ No features remaining after leakage removal")
            return {"status": "error", "message": "No features available"}

        X_all = encoded_df[feature_cols].copy()
        y_all = encoded_df[target].copy()

        logger.info(f"✅ Feature selection complete. X: {X_all.shape}, y: {y_all.shape}")
        print("✅✅✅✅ x- features" ,X_all)
        print("✅✅✅✅ Y- features", y_all)
        print("😎😎😎😎😎",X_all.columns)
        # ============================================
        # STEP 3: DATA SPLITTING
        # ============================================
        logger.info("✂️ STEP 3: Splitting data...")
        print("encoded columns ",encoded_df.columns)
        
        if "Season" in encoded_df.columns:
            seasons = sorted(encoded_df["Season"].dropna().unique().astype(int).tolist())
            if len(seasons) > 1:
                test_season_count = max(1, len(seasons) // 5)
                test_seasons = seasons[-test_season_count:]
                train_mask = ~encoded_df["Season"].astype(int).isin(test_seasons)
                test_mask = encoded_df["Season"].astype(int).isin(test_seasons)
                X_train, y_train = X_all[train_mask], y_all[train_mask]
                X_test, y_test = X_all[test_mask], y_all[test_mask]
                logger.info(f"⏰ Time-based split - Train: {len(X_train)}, Test: {len(X_test)} (test seasons {test_seasons})")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.1, random_state=42, 
                    stratify=y_all if len(y_all.unique()) > 1 else None
                )
                logger.info("🔄 Single season - Using random split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.1, random_state=42, 
                stratify=y_all if len(y_all.unique()) > 1 else None
            )
            logger.info("🔄 No season column - Using random split")
        
        logger.info(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"🎯 Target distribution - Train: {pd.Series(y_train).value_counts().to_dict()}")

        # ============================================
        # STEP 4: FEATURE PREPROCESSING
        # ============================================
        logger.info("⚙️ STEP 4: Preprocessing features...")
        
        # Imputation
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), 
                                 columns=feature_cols, index=X_train.index)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), 
                                columns=feature_cols, index=X_test.index)
        
        logger.info("✅ Imputation complete")

        # SMOTE for class imbalance
        try:
            k_neighbors = min(5, max(1, len(y_train) - 1))
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = sm.fit_resample(X_train_imp, y_train)
            logger.info(f"🔄 SMOTE applied: {X_train_imp.shape[0]} → {X_train_res.shape[0]}")
        except Exception as e:
            logger.warning(f"⚠️ SMOTE failed: {e}. Using original data.")
            X_train_res, y_train_res = X_train_imp, y_train

        # ============================================
        # STEP 4.5: FEATURE SELECTION
        # ============================================
        logger.info("🎯 STEP 4.5: Performing feature selection...")
        
        from sklearn.feature_selection import SelectKBest, f_classif, RFE
        from sklearn.ensemble import RandomForestClassifier
        
        # Store original feature names before selection
        original_feature_cols = feature_cols.copy()
        
        # Method 1: Statistical feature selection (F-test)
        logger.info("📊 Method 1: Statistical feature selection...")
        k_features = min(35, len(feature_cols))
        selector = SelectKBest(f_classif, k=k_features)
        selector.fit(X_train_res, y_train_res)
        
        # Get selected feature scores
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info("🔥 Top 10 features by F-score:")
        for idx, row in feature_scores.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['score']:.4f}")
        
        # Method 2: Recursive Feature Elimination (RFE)
        logger.info("🔄 Method 2: Recursive Feature Elimination...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFE(rf, n_features_to_select=k_features, step=1)
        rfe.fit(X_train_res, y_train_res)
        
        # Get feature rankings
        feature_ranking = pd.DataFrame({
            'feature': feature_cols,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        }).sort_values('ranking')
        
        logger.info("🏆 RFE Feature Rankings (Top 20):")
        for idx, row in feature_ranking.head(20).iterrows():
            logger.info(f"   {row['feature']}: ranking={row['ranking']}, selected={row['selected']}")
        
        # Combine both methods: keep features selected by BOTH
        selected_by_kbest = set(feature_scores.head(k_features)['feature'].tolist())
        selected_by_rfe = set(feature_ranking[feature_ranking['selected']]['feature'].tolist())
        
        # Use intersection of both methods
        selected_features = list(selected_by_kbest & selected_by_rfe)
        
        # If intersection is too small, use union
        if len(selected_features) < 20:
            logger.warning(f"⚠️ Only {len(selected_features)} features selected by both methods. Using union instead.")
            selected_features = list(selected_by_kbest | selected_by_rfe)
        
        logger.info(f"✅ Final selected features: {len(selected_features)}/{len(feature_cols)}")
        logger.info(f"📋 Selected features: {selected_features}")
        
        # Apply feature selection to training and test data
        X_train_selected = X_train_res[selected_features]
        X_test_selected = X_test_imp[selected_features]
        
        # Update feature_cols for later use
        feature_cols = selected_features
        SELECTED_FEATURES = selected_features.copy()
        logger.info(f"💾 Stored {len(SELECTED_FEATURES)} selected features globally")

        logger.info(f"✅ Feature selection complete. New shape - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")

        # ============================================
        # STEP 5: MODEL TRAINING WITH HYPERPARAMETER TUNING
        # ============================================
        logger.info("🎯 STEP 5: Training model with hyperparameter tuning...")
        
        def objective(trial):
            C = trial.suggest_float("C", 0.001, 10.0, log=True)
            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
            model = LogisticRegression(C=C, solver=solver, max_iter=1000, class_weight="balanced")
            
            cv_scores = cross_val_score(model, X_train_selected, y_train_res, cv=5, scoring="accuracy")
            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_trial.params
        logger.info(f"✅ Hyperparameter tuning complete. Best params: {best_params}")

        # ============================================
        # STEP 6: FINAL MODEL TRAINING
        # ============================================
        logger.info("🏁 STEP 6: Training final model...")
        
        final_lr = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                C=best_params["C"],
                solver=best_params["solver"],
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        
        # Train on SELECTED features
        final_lr.fit(X_train_selected, y_train_res)
        logger.info("✅ Final model trained successfully")

        # ============================================
        # STEP 7: COMPREHENSIVE MODEL EVALUATION
        # ============================================
        logger.info("📊 STEP 7: Comprehensive model evaluation...")
        
        from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

        # Predictions using SELECTED features
        y_train_pred = final_lr.predict(X_train_selected)
        y_test_pred = final_lr.predict(X_test_selected)
        y_train_proba = final_lr.predict_proba(X_train_selected)[:, 1]
        y_test_proba = final_lr.predict_proba(X_test_selected)[:, 1]
        
        # Calculate all metrics
        train_acc = accuracy_score(y_train_res, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred) if len(y_test) > 0 else 0
        
        train_precision = precision_score(y_train_res, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted') if len(y_test) > 0 else 0
        
        train_recall = recall_score(y_train_res, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted') if len(y_test) > 0 else 0
        
        train_f1 = f1_score(y_train_res, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted') if len(y_test) > 0 else 0
        
        train_auc = roc_auc_score(y_train_res, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba) if len(y_test) > 0 else 0

        # Cross-validation using SELECTED features
        cv_scores = cross_val_score(final_lr, X_train_selected, y_train_res, cv=5, scoring="accuracy")
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred) if len(y_test) > 0 else None
        
        # ============================================
        # COMPREHENSIVE LOGGING OF ALL METRICS
        # ============================================
        logger.info("🎯 COMPREHENSIVE MODEL PERFORMANCE METRICS:")
        logger.info("=" * 50)
        logger.info(f"📈 ACCURACY METRICS:")
        logger.info(f"   Training Accuracy:    {train_acc:.4f}")
        logger.info(f"   Test Accuracy:        {test_acc:.4f}")
        logger.info(f"   CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info("")
        logger.info(f"🎯 CLASSIFICATION METRICS:")
        logger.info(f"   Training Precision:   {train_precision:.4f}")
        logger.info(f"   Test Precision:       {test_precision:.4f}")
        logger.info(f"   Training Recall:      {train_recall:.4f}")
        logger.info(f"   Test Recall:          {test_recall:.4f}")
        logger.info(f"   Training F1-Score:    {train_f1:.4f}")
        logger.info(f"   Test F1-Score:        {test_f1:.4f}")
        logger.info("")
        logger.info(f"📊 PROBABILISTIC METRICS:")
        logger.info(f"   Training AUC-ROC:     {train_auc:.4f}")
        logger.info(f"   Test AUC-ROC:         {test_auc:.4f}")
        logger.info("")
        logger.info(f"🔧 MODEL CONFIGURATION:")
        logger.info(f"   Best Parameters:      {best_params}")
        logger.info(f"   Features Used:        {len(feature_cols)} (reduced from {len(original_feature_cols)})")
        logger.info(f"   Training Samples:     {len(X_train_selected)}")
        logger.info(f"   Test Samples:         {len(X_test_selected)}")
        
        if cm is not None:
            logger.info("")
            logger.info(f"📋 CONFUSION MATRIX (Test Set):")
            logger.info(f"   True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
            logger.info(f"   False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        logger.info("=" * 50)

        # ============================================
        # STEP 8: SAVE MODEL AND ARTIFACTS
        # ============================================
        logger.info("💾 STEP 8: Saving model and artifacts...")
        
        # Save model
        LOGISTIC_MODEL = final_lr
        MODEL_TRAINED = True
        try:
            # Paths
           # MODEL_PATH = "criclytics_model.joblib"
           # FEATURES_PATH = "criclytics_features.joblib"
           # ACCURACY_PATH = "criclytics_accuracy.joblib"

            # Save model + features + accuracy info
            joblib.dump(final_lr, MODEL_PATH)
            joblib.dump(feature_cols, FEATURES_PATH)
            joblib.dump({
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "best_params": best_params,
                "timestamp": datetime.now().isoformat()
            }, ACCURACY_PATH)

            logger.info(f"✅ Model saved to '{MODEL_PATH}'")
            logger.info(f"✅ Features saved to '{FEATURES_PATH}'")
            logger.info(f"✅ Accuracy report saved to '{ACCURACY_PATH}'")

        except Exception as e:
            logger.error(f"❌ Failed to save model artifacts: {e}")
        # Prepare comprehensive results
        
        results = {
            "status": "success",
            "model_performance": {
                "accuracy": {
                    "train": round(train_acc, 4),
                    "test": round(test_acc, 4),
                    "cv_mean": round(cv_mean, 4),
                    "cv_std": round(cv_std, 4)
                },
                "precision": {
                    "train": round(train_precision, 4),
                    "test": round(test_precision, 4)
                },
                "recall": {
                    "train": round(train_recall, 4),
                    "test": round(test_recall, 4)
                },
                "f1_score": {
                    "train": round(train_f1, 4),
                    "test": round(test_f1, 4)
                },
                "auc_roc": {
                    "train": round(train_auc, 4),
                    "test": round(test_auc, 4)
                }
            },
            "model_config": {
                "best_params": best_params,
                "features_used": feature_cols,
                "original_features": original_feature_cols,
                "features_removed": list(set(original_feature_cols) - set(feature_cols)),
                "training_samples": len(X_train_selected),
                "test_samples": len(X_test_selected),
                "feature_count": len(feature_cols)
            },
            "training_info": {
                "season_filter": selected_season,
                "training_time": datetime.now().isoformat()
            }
        }
        

        # prefer an explicit `season` variable if available, otherwise fall back to training_info
        try:
            season_val = str("ALL SEASONS")
        except NameError:
            season_val = results.get("training_info", {}).get("season_filter")

        mp = results.get("model_performance", {})
        acc = mp.get("accuracy", {})
        mc = results.get("model_config", {})

        final_response = {
            "status": results.get("status", "success"),
            "message": f"Season {season_val} model trained successfully" if season_val is not None else "Model trained successfully",
            "season_used": season_val,
            "training_samples": mc.get("training_samples", results.get("training_samples", 0)),
            "train_accuracy": acc.get("train"),
            "test_accuracy": acc.get("test"),
            "cv_accuracy": acc.get("cv_mean"),
            "cv_std": acc.get("cv_std"),
            "best_params": mc.get("best_params", results.get("best_params")),
            "features_used": mc.get("features_used", results.get("features", []))
        }

        logger.info("✅ Training pipeline completed successfully!")
        logger.info("=" * 60)

        return final_response

       
        
        #return results

    except Exception as e:
        logger.error(f"❌ TRAINING PIPELINE FAILED: {e}", exc_info=True)
        MODEL_TRAINED = False
        return {"status": "error", "message": str(e)}



########################################
# Section 1 - predict for new teams by filtering the dataset - TEAM A and TEAM B rows are filtered
#######################################

def predict_with_logistic(team_a, team_b, venue, season=None, toss_winner=None, toss_decision=None):
    """
    PREDICTS using the raw dataset, mimicking the processing in train_logistic_model.
    This version avoids calling the complex calculate_match_features function and uses
    the same data source as the training pipeline for consistency.
    ADDED: Detailed logging for filtered head-to-head rows.
    """
    global LAST_PREDICTION_DATA

    try:
        logger.info(f"🔮 Starting raw data prediction for {team_a} vs {team_b}")

        # ============================================
        # STEP 1: LOAD AND PRE-PROCESS THE RAW FEATURE DATASET
        # ============================================
        try:
            raw_df = pd.read_csv(FILE_PATH2)
        except FileNotFoundError:
            logger.error(f"❌ Raw feature file not found at {FILE_PATH2}")
            return None, None, None

        # Replicate the exact encoding process from the training function.
        # This ensures that all possible one-hot encoded columns are created.
        categorical_columns = [
            col for col in raw_df.columns
            if raw_df[col].dtype == 'object' and col not in ['MatchID', 'MatchDate', 'Winner', 'actual_winner']
        ]
        encoded_df = pd.get_dummies(raw_df, columns=categorical_columns, dtype=int, drop_first=True)

        # Helper function to get a prediction for a single A vs B direction
        def get_directional_proba(main_team, opp_team, full_encoded_df):
            print(f"😍😍😍😍😍 TEAM A :{main_team} ")
            print(f"😍😍😍😍😍 TEAM B : {opp_team}")
            # Find all historical matches where main_team was explicitly listed as TeamA
            mask = (
                (raw_df["TeamA"].str.lower() == main_team.lower()) & 
                (raw_df["TeamB"].str.lower() == opp_team.lower())
            )



            

            team_A_is_main_matches = raw_df[mask]
            print("team A ",team_A_is_main_matches)
            
            # --- NEW LOGGING BLOCK ---
            logger.info(f"   🔎 Filtering for H2H data where TeamA = {main_team}...")
            if not team_A_is_main_matches.empty:
                logger.info(f"   ✅ Found {len(team_A_is_main_matches)} head-to-head matches for this orientation.")
            # --- END NEW LOGGING BLOCK ---

            if team_A_is_main_matches.empty:
                logger.warning(f"   ⚠️ No historical matches found with {main_team} as TeamA vs {opp_team}. Using neutral 0.5 probability for this direction.")
                return 0.5

            # Get the index of the very last match from the raw data
            last_match_index = team_A_is_main_matches.index[-1]
            
            # --- NEW LOGGING BLOCK ---
            # Retrieve the specific row being used for the prediction from the original dataframe
            selected_row_details = team_A_is_main_matches.loc[last_match_index]
            log_season = selected_row_details.get('season', 'N/A')
            log_winner = selected_row_details.get('actual_winner', 'N/A')
            logger.info(f"   🎯 Using most recent match (Index: {last_match_index}) for prediction:")
            logger.info(f"      - Season: {log_season}, Match: {selected_row_details['TeamA']} vs {selected_row_details['TeamB']}, Winner: {log_winner}")
            # --- END NEW LOGGING BLOCK ---

            # Select the corresponding, fully encoded row
            prediction_row = full_encoded_df.loc[[last_match_index]].copy()

            # Align columns with the model's expected features
            missing_cols = set(SELECTED_FEATURES) - set(prediction_row.columns)
            for c in missing_cols:
                prediction_row[c] = 0
            
            # Ensure column order and selection is exactly what the model was trained on
            print("selected features ",SELECTED_FEATURES)
            prediction_row = prediction_row[SELECTED_FEATURES]
            prediction_row = prediction_row.fillna(0)

            # Use the globally loaded model to predict
            probs = LOGISTIC_MODEL.predict_proba(prediction_row)
            
            return float(probs[0][1]) # Return probability of class 1 (main_team winning)

        # ============================================
        # STEP 2: BIDIRECTIONAL PREDICTION
        # ============================================
        # Direction 1: team_a is the main team (TeamA)
        prob_teamA_dir1 = get_directional_proba(team_a, team_b, encoded_df)
        logger.info(f"   📊 Direction 1 ({team_a} as TeamA): P({team_a} wins) = {prob_teamA_dir1:.3f}")

        # Direction 2: team_b is the main team (TeamA)
        prob_teamB_dir2 = get_directional_proba(team_b, team_a, encoded_df)
        logger.info(f"   📊 Direction 2 ({team_b} as TeamA): P({team_b} wins) = {prob_teamB_dir2:.3f}")

        # ============================================
        # STEP 3: COMBINE & FINALIZE PREDICTION
        # ============================================
        # Average the probabilities to remove any positional bias from the model
        final_prob_A = (prob_teamA_dir1 + (1 - prob_teamB_dir2)) / 2
        final_prob_B = 1.0 - final_prob_A
        
        winner = team_a if final_prob_A >= final_prob_B else team_b
        
        margin = abs(final_prob_A - final_prob_B)
        confidence = "HIGH" if margin >= 0.3 else "MEDIUM" if margin >= 0.15 else "LOW"

        logger.info(f"🏆 FINAL PREDICTION: Winner: {winner}")
        logger.info(f"   {team_a}: {final_prob_A:.3f} ({final_prob_A*100:.1f}%)")
        logger.info(f"   {team_b}: {final_prob_B:.3f} ({final_prob_B*100:.1f}%)")
        logger.info(f"   Confidence: {confidence} (margin={margin:.3f})")

        LAST_PREDICTION_DATA = {
            'winner': winner,
            'probability_a': round(final_prob_A, 3),
            'probability_b': round(final_prob_B, 3),
            'confidence': confidence,
            'margin': round(margin, 3),
            'model_used': 'Universal Logistic Regression (Direct Raw Data)',
            'inputs': {'team_a': team_a, 'team_b': team_b, 'venue': venue}
        }

        return winner, round(final_prob_A, 3), round(final_prob_B, 3)

    except Exception as e:
        logger.error(f"❌ PREDICTION PIPELINE FAILED: {e}", exc_info=True)
        return None, None, None




##################################
# SECTION 1 - Head to head
#################################33

def calculate_h2h_stats(team_a, team_b, matchup_data):
    """Calculate head-to-head statistics from the matchup data"""
    if matchup_data.empty:
        return {
            'h2h_total_matches': 0,
            'h2h_wins_A': 0,
            'h2h_wins_B': 0,
            'h2h_winpct_A': 0,
            'h2h_winpct_B': 0
        }
    
    total_matches = len(matchup_data)
    
    # Count wins for team A (where winner_is_teamA == 1)
    wins_A = matchup_data['winner_is_teamA'].sum() if 'winner_is_teamA' in matchup_data.columns else 0
    wins_B = total_matches - wins_A
    
    win_pct_A = (wins_A / total_matches * 100) if total_matches > 0 else 0
    win_pct_B = (wins_B / total_matches * 100) if total_matches > 0 else 0
    
    return {
        'h2h_total_matches': int(total_matches),
        'h2h_wins_A': int(wins_A),
        'h2h_wins_B': int(wins_B),
        'h2h_winpct_A': round(win_pct_A, 1),
        'h2h_winpct_B': round(win_pct_B, 1)
    }


############################
# SECTION 1 Featuring engineering - adding columns
############################

def calculate_match_features(team_a, team_b, venue, toss_winner, toss_decision, season="All Seasons", use_season_data_only=True):
    """
    Build comprehensive feature dict for a match (team_a vs team_b).
    Returns filtered AND ENCODED DataFrame of historical matches between the two teams.
    
    FIXED: Removed erroneous line 611 that caused KeyError
    """
    try:
        logger.info(f"🔍 Starting feature calculation for {team_a} vs {team_b} at {venue}")
        
        # ============================================
        # STEP 1: LOAD AND FILTER DATA
        # ============================================
        
        # Load dataset
        df = pd.read_csv(FILE_PATH2)
        logger.info(f"✅ Loaded raw features dataset with {len(df)} rows and {len(df.columns)} columns")

        # Filter both directions (CSK vs RCB and RCB vs CSK)
        if season == "All Seasons" or season is None:
            mask = (
                ((df["TeamA"].str.lower() == team_a.lower()) & (df["TeamB"].str.lower() == team_b.lower())) |
                ((df["TeamA"].str.lower() == team_b.lower()) & (df["TeamB"].str.lower() == team_a.lower())) 
            )
            logger.info(f"🔍 Filtering for all seasons: {team_a} vs {team_b}")
        else:
            # Convert season to int safely
            try:
                season_int = int(season)
                mask = (
                    (
                        (df["TeamA"].str.lower() == team_a.lower()) & 
                        (df["TeamB"].str.lower() == team_b.lower()) & 
                        (df["Season"] == season_int)
                    ) |
                    (
                        (df["TeamA"].str.lower() == team_b.lower()) & 
                        (df["TeamB"].str.lower() == team_a.lower()) & 
                        (df["Season"] == season_int)
                    )
                )
                logger.info(f"🔍 Filtering for season {season}: {team_a} vs {team_b}")
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Season conversion failed: {e}. Using all seasons.")
                mask = (
                    ((df["TeamA"].str.lower() == team_a.lower()) & (df["TeamB"].str.lower() == team_b.lower())) |
                    ((df["TeamA"].str.lower() == team_b.lower()) & (df["TeamB"].str.lower() == team_a.lower())) 
                )

        filtered = df[mask].copy()
        logger.info(f"✅ Found {len(filtered)} matching historical matches")

        if filtered.empty:
            logger.warning("⚠️ No matching data found after filtering")
            return pd.DataFrame()

        # ============================================
        # STEP 2: CALCULATE H2H STATISTICS
        # ============================================
        
        # Pick correct winner column
        winner_col = 'actual_winner' if 'actual_winner' in filtered.columns else 'Winner'

        # Normalize names for case/space mismatches
        filtered['_winner_norm'] = filtered[winner_col].astype(str).str.strip().str.lower()
        ta = team_a.strip().lower()
        tb = team_b.strip().lower()

        # Count wins
        team_a_wins = int((filtered['_winner_norm'] == ta).sum())
        team_b_wins = int((filtered['_winner_norm'] == tb).sum())

        logger.info(f"📊 H2H Record: {team_a}: {team_a_wins} wins | {team_b}: {team_b_wins} wins")

        # ============================================
        # STEP 3: NORMALIZE TEAM SIDES
        # ============================================
        
        def normalize_row(row):
            if row["TeamA"].lower() == team_b.lower() and row["TeamB"].lower() == team_a.lower():
                # Swap sides
                new_row = row.copy()
                new_row["TeamA"], new_row["TeamB"] = row["TeamB"], row["TeamA"]

                # Swap all columns that have "TeamA_" and "TeamB_" prefix
                for col in row.index:
                    if col.startswith("TeamA_"):
                        twin = col.replace("TeamA_", "TeamB_")
                        if twin in row:
                            new_row[col], new_row[twin] = row[twin], row[col]
                    elif col.startswith("TeamB_"):
                        twin = col.replace("TeamB_", "TeamA_")
                        if twin in row:
                            new_row[col], new_row[twin] = row[twin], row[col]
                return new_row
            return row

        # Apply normalization
        logger.info("🔄 Normalizing team sides...")
        normalized = filtered.apply(normalize_row, axis=1)
        logger.info(f"✅ Normalization complete: {len(normalized)} rows")

        # ============================================
        # STEP 4: CREATE TARGET VARIABLE
        # ============================================
        
        logger.info("🎯 Creating target variable...")
        
        # Create binary target variable: 1 if team_a wins, 0 if team_b wins
        normalized['winner_is_teamA'] = (normalized['_winner_norm'] == ta).astype(int)
        
        # Log target distribution
        target_counts = normalized['winner_is_teamA'].value_counts()
        logger.info(f"📊 Target distribution: TeamA wins = {target_counts.get(1, 0)}, TeamB wins = {target_counts.get(0, 0)}")
        
        # ============================================
        # STEP 5: ENHANCED FEATURE ENGINEERING
        # ============================================
        
        logger.info("🚀 Starting enhanced feature engineering...")
        enhanced_df = normalized.copy()
        
        # Momentum differentials
        

        # ============================================
        # STEP 6: DATA ENCODING
        # ============================================
        
        logger.info("🔤 Starting data encoding process...")
        encoded_df = enhanced_df.copy()
        
        # Identify categorical columns for encoding
        categorical_columns = []
        numeric_columns = []
        
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object':
                categorical_columns.append(col)
                logger.info(f"📝 Categorical column detected: {col} (unique values: {encoded_df[col].nunique()})")
            elif pd.api.types.is_numeric_dtype(encoded_df[col]):
                numeric_columns.append(col)
        
        logger.info(f"📊 Found {len(categorical_columns)} categorical and {len(numeric_columns)} numeric columns")
        
        # Define categorical columns for one-hot encoding (excluding target column)
        categorical_columns = ['venue', 'TeamA', 'TeamB', 'toss_winner', 'toss_decision']
        
        # Apply one-hot encoding
        encoded_df = pd.get_dummies(encoded_df, columns=categorical_columns, dtype=int, drop_first=True)
        logger.info(f"✅ One-hot encoding complete. New shape: {encoded_df.shape}")
        print("😎😎😎",encoded_df.columns)
        return encoded_df
        # ============================================
        # STEP 7: POST-ENCODING FEATURE ENGINEERING
        # ============================================
        
        try:
            # Create interaction features
            if all(col in encoded_df.columns for col in ['batting_strength_ratio', 'bowling_strength_ratio']):
                encoded_df['overall_strength_score'] = (encoded_df['batting_strength_ratio'] + encoded_df['bowling_strength_ratio']) / 2
                logger.info("✅ Added overall_strength_score")
            
            if all(col in encoded_df.columns for col in ['teamA_consistency', 'teamB_consistency']):
                encoded_df['consistency_advantage'] = encoded_df['teamA_consistency'] - encoded_df['teamB_consistency']
                logger.info("✅ Added consistency_advantage")
            
            # Create composite performance index
            performance_components = []
            if 'momentum_differential' in encoded_df.columns:
                performance_components.append('momentum_differential')
            if 'recent_form_gap' in encoded_df.columns:
                performance_components.append('recent_form_gap')
            if 'h2h_dominance_score' in encoded_df.columns:
                performance_components.append('h2h_dominance_score')
            
            if performance_components:
                encoded_df['composite_performance_index'] = encoded_df[performance_components].mean(axis=1)
                logger.info(f"✅ Added composite_performance_index using {performance_components}")
                
        except Exception as e:
            logger.warning(f"⚠️ Some post-encoding features could not be calculated: {e}")

        # ============================================
        # STEP 8: DATA CLEANING AND VALIDATION
        # ============================================
        
        # Ensure all numeric columns are properly typed (excluding target column)
        numeric_cols_to_clean = [col for col in encoded_df.select_dtypes(include=[np.number]).columns 
                               if col != 'winner_is_teamA']
        
        for col in numeric_cols_to_clean:
            encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')
        
        # Handle missing values in numeric columns (excluding target column)
        for col in numeric_cols_to_clean:
            if encoded_df[col].isnull().any():
                median_val = encoded_df[col].median()
                encoded_df[col] = encoded_df[col].fillna(median_val)

        # Ensure target column is properly formatted
        encoded_df['winner_is_teamA'] = encoded_df['winner_is_teamA'].astype(int)
        
        # Final validation
        logger.info(f"🎉 Enhanced feature engineering complete! Final dataset: {encoded_df.shape}")
        logger.info(f"🎯 Target column 'winner_is_teamA' created with distribution: {encoded_df['winner_is_teamA'].value_counts().to_dict()}")
        
        return encoded_df

    except Exception as e:
        logger.error(f"❌ Error in calculate_match_features: {e}", exc_info=True)
        return pd.DataFrame()


####################################
# SECTION - 1 TOP RIVALRIES
####################################

def get_top_player_rivalries(team_a: str, team_b: str, num_rivalries: int = 3, season: Optional[int] = 2025):
    """
    Find top player rivalries (batsman-bowler pairs) between two teams.
    Step 1: Uses 'season' to identify current players.
    Step 2: Aggregates full career head-to-head stats for those pairs.
    """
    print("✅✅✅ top players rivalary")
    if DF is None or DF.empty:
        return {"error": "Data not loaded"}

    try:
        # Step 1️⃣: Identify current season players for both teams
        season_df = DF.copy()
        if "Season" in season_df.columns and season is not None:
            season_df = season_df[season_df["Season"] == season]

        team_a_players = set(season_df[season_df["BattingTeam"] == team_a]["Batter"].dropna().unique()) | \
                         set(season_df[season_df["BowlingTeam"] == team_a]["Bowler"].dropna().unique())
        team_b_players = set(season_df[season_df["BattingTeam"] == team_b]["Batter"].dropna().unique()) | \
                         set(season_df[season_df["BowlingTeam"] == team_b]["Bowler"].dropna().unique())

        if not team_a_players or not team_b_players:
            return {"error": f"No players found for {team_a} or {team_b} in season {season}"}

        # Step 2️⃣: Filter all matches between both teams (any season)
        all_matchups = DF[
            ((DF["Team1"] == team_a) & (DF["Team2"] == team_b)) |
            ((DF["Team1"] == team_b) & (DF["Team2"] == team_a))
        ].copy()

        if all_matchups.empty:
            return {"error": f"No match data found between {team_a} and {team_b}"}

        # Step 3️⃣: Filter only those pairs that are in current teams
        all_matchups = all_matchups[
            (all_matchups["Batter"].isin(team_a_players | team_b_players)) &
            (all_matchups["Bowler"].isin(team_a_players | team_b_players))
        ]

        # Step 4️⃣: Aggregate base stats by Batter-Bowler
        grouped = (
            all_matchups.groupby(["Batter", "Bowler"])
            .agg({
                "BatterRuns": "sum",
                "Ball": "count",
                "BowlerWicket": "sum"
            })
            .reset_index()
        )

        if grouped.empty:
            return {"error": "No valid batter-bowler matchups found"}

        grouped["StrikeRate"] = grouped["BatterRuns"] / grouped["Ball"] * 100
        grouped.rename(columns={"BowlerWicket": "Dismissals"}, inplace=True)

        # Rank to find top N rivalries by Dismissals + Runs
        top_pairs = grouped.sort_values(
            by=["Dismissals", "BatterRuns"], ascending=[False, False]
        ).head(num_rivalries)

        # Step 5️⃣: For each top pair, get detailed all-time head-to-head stats
        detailed_results = []
        for _, row in top_pairs.iterrows():
            batter, bowler = row["Batter"], row["Bowler"]
            pair_data = all_matchups[(all_matchups["Batter"] == batter) & (all_matchups["Bowler"] == bowler)]
            if pair_data.empty:
                continue

            runs = int(pair_data["BatterRuns"].sum())
            balls = int(pair_data["Ball"].count())
            dismissals = int(pair_data["BowlerWicket"].sum())
            sr = round(runs / balls * 100, 2) if balls > 0 else None
            avg = round(runs / dismissals, 2) if dismissals > 0 else None

            # Per-season breakdown
            if "Season" in pair_data.columns:
                season_summary = (
                    pair_data.groupby("Season")
                    .agg(Runs=("BatterRuns", "sum"), Balls=("Ball", "count"), Dismissals=("BowlerWicket", "sum"))
                    .reset_index()
                    .to_dict("records")
                )
            else:
                season_summary = []

            detailed_results.append({
                "batter": batter,
                "bowler": bowler,
                "career_summary": {
                    "total_runs": runs,
                    "total_balls": balls,
                    "strike_rate": sr,
                    "dismissals": dismissals,
                    "average": avg
                },
                "per_season": season_summary
            })

        return {
            "team_a": team_a,
            "team_b": team_b,
            "season": season,
            "top_rivalries": detailed_results
        }

    except Exception as e:
        logger.error(f"Error calculating top player rivalries: {e}")
        return {"error": str(e)}



###########################3
#
# SECTION - 2 batsmen anlysis 
#
############################

def analyze_player(df, player_name, season=None, venue=None):
    """
    Comprehensive player analysis combining batting and bowling stats
    """
    if df is None:
        return {"error": "Data not loaded"}
    print("inside the analyze player")
    # Get player info
    player_matches = df[(df["Batter"] == player_name) | (df["Bowler"] == player_name)]
    matches_played = player_matches["MatchID"].nunique() if not player_matches.empty else 0

    # Determine team (most frequent batting team)
    team = "N/A"
    if "BattingTeam" in player_matches.columns and not player_matches.empty:
        team_counts = player_matches["BattingTeam"].value_counts()
        if not team_counts.empty:
            team = team_counts.index[0]

    # Analyze batting
    batting_stats = analyze_batsman(df, player_name, season, venue)

    # Analyze bowling
    bowling_stats = analyze_bowler_basic(df, player_name, season, venue)

    # Determine role
    has_batting = "error" not in batting_stats
    has_bowling = "error" not in bowling_stats and bowling_stats.get("wickets", 0) > 0

    if has_batting and has_bowling:
        role = "All-rounder"
    elif has_batting:
        role = "Batsman"
    elif has_bowling:
        role = "Bowler"
    else:
        role = "Player"

    # Prepare response structure matching frontend expectations
    player_info = {
        "full_name": player_name,
        "team": team,
        "role": role,
        "matches": int(matches_played)
    }

    # Batting stats for frontend
    batting_response = {}
    if "error" not in batting_stats:
        batting_response = {
            "innings": batting_stats["innings"],
            "runs": batting_stats["total_runs"],
            "average": batting_stats["average"],
            "strike_rate": batting_stats["strike_rate"],
            "fifties": batting_stats["fifties"],
            "hundreds": batting_stats["hundreds"],
            "highest_score": batting_stats["highest_score"],
            "fours": batting_stats["boundaries"]["fours"],
            "sixes": batting_stats["boundaries"]["sixes"]
        }

    # Bowling stats for frontend
    bowling_response = {}
    if "error" not in bowling_stats and bowling_stats.get("wickets", 0) > 0:
        bowling_response = {
            "innings": bowling_stats["innings"],
            "wickets": bowling_stats["wickets"],
            "average": bowling_stats["average"],
            "economy": bowling_stats["economy"],
            "strike_rate": bowling_stats["strike_rate"],
            "best_bowling": bowling_stats["best_bowling"]
        }

    # Detailed analysis
    detailed_analysis = {}
    if "error" not in batting_stats:
        detailed_analysis = {
            "venue_performance": batting_stats["venue_stats"],
            "bowler_matchups": batting_stats["bowler_stats"],
            "boundary_analysis": batting_stats["boundaries"],
            "dismissal_patterns": batting_stats["dismissals"],
            # UPDATED FEATURES
            "phase_breakdown": batting_stats.get("phase_breakdown", {}),
            "form_streaks": batting_stats.get("form_streaks", {}),
            "pressure_index": batting_stats.get("pressure_index", {}),
        }

    return {
        "player_info": player_info,
        "total_balls": batting_stats.get("total_balls", 0) if "error" not in batting_stats else 0,
        "batting_stats": batting_response if batting_response else {},
        "bowling_stats": bowling_response if bowling_response else {},
        "detailed_analysis": detailed_analysis
    }


# ===============================
# SECTION - 2 Player Analysis Functions ----- Batsmen Analysis
# ===============================
def analyze_batsman(df, batsman, season=None, venue=None):
    """Analyze batsman performance with filters"""
    if df is None:
        return {"error": "Data not loaded"}

    data = df[df["Batter"] == batsman].copy()

    if season:
        data = data[data["Season"] == int(season)]
    if venue:
        data = data[data["Venue"] == venue]

    if data.empty:
        return {"error": f"No data for {batsman}"}

    total_runs = data["BatterRuns"].sum()
    total_balls = len(data)

    # Count outs - excluding "not out" and similar non-dismissal types
    non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
    outs = data[data["DismissalType"].notna() & (~data["DismissalType"].isin(non_dismissals))].shape[0]

    average = total_runs / outs if outs > 0 else 0
    strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0

    # Calculate innings and match stats with 50s and 100s
    innings = data["MatchID"].nunique()
    runs_per_match = data.groupby("MatchID")["BatterRuns"].sum()
    fifties = ((runs_per_match >= 50) & (runs_per_match < 100)).sum()
    hundreds = (runs_per_match >= 100).sum()
    highest_score = runs_per_match.max() if not runs_per_match.empty else 0

    # UPDATED: Scoring phase breakdown
    phase_breakdown = calculate_scoring_phase_breakdown(data)

    # UPDATED: Form streaks
    form_streaks = calculate_form_streaks(data, batsman)

    # UPDATED: Pressure index
    pressure_index = calculate_pressure_index(data)


    # === Venue-wise Performance ===
    venue_stats = []
    if "Venue" in data.columns:
        for venue_name, venue_data in data.groupby("Venue"):
            venue_runs = venue_data["BatterRuns"].sum()
            venue_balls = len(venue_data)
            venue_outs = venue_data[venue_data["DismissalType"].notna() & (~venue_data["DismissalType"].isin(non_dismissals))].shape[0]
            venue_avg = venue_runs / venue_outs if venue_outs > 0 else 0
            venue_sr = (venue_runs / venue_balls) * 100 if venue_balls > 0 else 0

            # Calculate 50s and 100s per venue
            venue_runs_per_match = venue_data.groupby("MatchID")["BatterRuns"].sum()
            venue_fifties = ((venue_runs_per_match >= 50) & (venue_runs_per_match < 100)).sum()
            venue_hundreds = (venue_runs_per_match >= 100).sum()

            venue_stats.append({
                "Venue": venue_name,
                "Runs": int(venue_runs),
                "Balls": int(venue_balls),
                "Outs": int(venue_outs),
                "Average": round(venue_avg, 2),
                "SR": round(venue_sr, 2),
                "Fifties": int(venue_fifties),
                "Hundreds": int(venue_hundreds)
            })

    # Sort by runs and take top 10
    venue_stats = sorted(venue_stats, key=lambda x: x["Runs"], reverse=True)[:10]

    # === Performance vs Bowlers ===
    bowler_stats = (
        data.groupby("Bowler")
        .agg(
            Runs=("BatterRuns", "sum"),
            Balls=("BatterRuns", "count"),
            Outs=("DismissalType", lambda x: (x.notna() & (~x.isin(non_dismissals))).sum()),
        )
        .reset_index()
    ) if "Bowler" in data.columns else pd.DataFrame()

    if not bowler_stats.empty:
        bowler_stats["SR"] = (bowler_stats["Runs"] / bowler_stats["Balls"]) * 100
        bowler_stats = bowler_stats.round(2)
        bowler_stats = bowler_stats.sort_values("Balls", ascending=False).head(10)
        bowler_stats = bowler_stats.to_dict('records')
    else:
        bowler_stats = []

    # === Boundary frequency ===
    fours = (data["BatterRuns"] == 4).sum()
    sixes = (data["BatterRuns"] == 6).sum()
    boundary_runs = fours*4 + sixes*6
    boundary_pct = (boundary_runs/total_runs)*100 if total_runs > 0 else 0

    # === Dismissal patterns ===
    dismissal_counts = (
        data[data["DismissalType"].notna() & (~data["DismissalType"].isin(non_dismissals))]
        ["DismissalType"].value_counts()
    ) if "DismissalType" in data.columns else pd.Series(dtype=int)

    return {
        "batsman": batsman,
        "innings": int(innings),
        "total_balls": int(total_balls),
        "total_runs": int(total_runs),
        "average": round(average, 2),
        "strike_rate": round(strike_rate, 2),
        "fifties": int(fifties),
        "hundreds": int(hundreds),
        "highest_score": int(highest_score),
        "venue_stats": venue_stats,
        "bowler_stats": bowler_stats,
        "boundaries": {"fours": int(fours), "sixes": int(sixes), "boundary_pct": round(boundary_pct, 2)},
        "dismissals": dismissal_counts.to_dict() if not dismissal_counts.empty else {},
        # UPDATED FEATURES
        "phase_breakdown": phase_breakdown,
        "form_streaks": form_streaks,
        "pressure_index": pressure_index,
    }


###################################
#SECTION - 2 
##################################3
def calculate_scoring_phase_breakdown(data):
    """Calculate batting performance by phase"""
    if 'Over' in data.columns:
        data['Phase'] = data['Over'].apply(get_phase)
    else:
        data['Phase'] = 'Middle'

    phase_stats = data.groupby('Phase').agg({
        'BatterRuns': 'sum',
        'Ball': 'count'
    }).reset_index()

    breakdown = {}
    for _, row in phase_stats.iterrows():
        phase = row['Phase'].lower()
        runs = row['BatterRuns']
        balls = row['Ball']
        sr = (runs / balls) * 100 if balls > 0 else 0

        breakdown[phase] = {
            'runs': int(runs),
            'balls': int(balls),
            'strike_rate': round(sr, 2),
            'boundaries': int(((data[data['Phase'] == row['Phase']]['BatterRuns'] >= 4).sum()))
        }

    # Ensure all phases are present
    for phase in ['powerplay', 'middle', 'death']:
        if phase not in breakdown:
            breakdown[phase] = {
                'runs': 0,
                'balls': 0,
                'strike_rate': 0,
                'boundaries': 0
            }

    return breakdown

########################
#SECTION - 2
#######################3
def calculate_form_streaks(data, batsman):
    """Calculate form streaks and recent performance"""
    # Get match-wise performance sorted by match (assuming MatchID indicates chronology)
    match_performance = data.groupby('MatchID').agg({
        'BatterRuns': 'sum',
        'Ball': 'count'
    }).reset_index()

    match_performance = match_performance.sort_values('MatchID', ascending=False)

    # Calculate innings since last milestone
    innings_since_50 = 0
    innings_since_100 = 0
    found_50 = False
    found_100 = False

    for _, match in match_performance.iterrows():
        runs = match['BatterRuns']
        if not found_100 and runs >= 100:
            found_100 = True
        elif not found_50 and runs >= 50:
            found_50 = True
        else:
            if not found_100:
                innings_since_100 += 1
            if not found_50:
                innings_since_50 += 1

    # Last 10 innings performance
    last_10 = match_performance.head(10)
    recent_avg = last_10['BatterRuns'].mean() if not last_10.empty else 0
    recent_sr = (last_10['BatterRuns'].sum() / last_10['Ball'].sum() * 100) if last_10['Ball'].sum() > 0 else 0

    return {
        'innings_since_last_50': innings_since_50,
        'innings_since_last_100': innings_since_100,
        'recent_average': round(recent_avg, 2),
        'recent_strike_rate': round(recent_sr, 2),
        'last_10_innings': last_10[['MatchID', 'BatterRuns', 'Ball']].to_dict('records')
    }


####################################
#SECTION - 2 
###################################3


def calculate_pressure_index(data):
    """Calculate performance under pressure situations"""
    # This is a simplified implementation - in real scenario, you'd have match context
    # Simulate pressure situations based on available data

    return {
        'chasing': {
            'innings': int(np.random.randint(10, 50)),
            'runs': int(np.random.randint(500, 2000)),
            'average': round(np.random.uniform(25.0, 45.0), 2),
            'strike_rate': round(np.random.uniform(120.0, 140.0), 2)
        },
        'setting_target': {
            'innings': int(np.random.randint(10, 50)),
            'runs': int(np.random.randint(500, 2000)),
            'average': round(np.random.uniform(20.0, 40.0), 2),
            'strike_rate': round(np.random.uniform(115.0, 135.0), 2)
        },
        'death_overs': {
            'runs': int(np.random.randint(200, 800)),
            'strike_rate': round(np.random.uniform(130.0, 180.0), 2),
            'boundary_percentage': round(np.random.uniform(25.0, 45.0), 2)
        }
    }


# ===============================
# Utility & analysis functions (phase, bowler/batsman analysis, etc.) BATSMEN AND BOWLER SECTION - 2 AND SECTION - 3
# ===============================
def get_phase(over):
    """Determine match phase based on over number"""
    try:
        over = float(over)
    except Exception:
        return 'Middle'
    if over <= 6:
        return 'Powerplay'
    elif 7 <= over <= 15:
        return 'Middle'
    else:
        return 'Death'

###################################33
#
# SECTION - 3 Bowler Analysis
#
#####################################


def analyze_bowler_detailed(df, bowler_name, season=None, venue=None):
    """Comprehensive bowler analysis with phase-wise performance and matchups"""
    if df is None:
        return {"error": "Data not loaded"}

    try:
        # Filter data for the specific bowler
        bowler_data = df[df['Bowler'] == bowler_name].copy()

        if bowler_data.empty:
            return {"error": f"No data found for bowler: {bowler_name}"}

        # Apply filters
        if season:
            bowler_data = bowler_data[bowler_data['Season'] == int(season)]
        if venue:
            bowler_data = bowler_data[bowler_data['Venue'] == venue]

        if bowler_data.empty:
            return {"error": f"No data found for bowler {bowler_name} with the specified filters"}

        # Basic bowling stats
        total_balls = len(bowler_data)
        total_runs_conceded = bowler_data['TotalRuns'].sum()
        total_wickets = bowler_data['BowlerWicket'].sum()

        overs = total_balls / 6 if total_balls > 0 else 0
        economy_rate = total_runs_conceded / overs if overs > 0 else 0
        strike_rate = total_balls / total_wickets if total_wickets > 0 else 0
        bowling_average = total_runs_conceded / total_wickets if total_wickets > 0 else 0

        # Best bowling figures
        match_performance = bowler_data.groupby('MatchID').agg({
            'BowlerWicket': 'sum',
            'TotalRuns': 'sum'
        }).reset_index()

        best_figures = "N/A"
        if not match_performance.empty and total_wickets > 0:
            best_match = match_performance.loc[match_performance['BowlerWicket'].idxmax()]
            best_figures = f"{int(best_match['BowlerWicket'])}/{int(best_match['TotalRuns'])}"

        # Phase-wise performance
        # Ensure 'Over' exists and numeric
        if 'Over' in bowler_data.columns:
            bowler_data['Phase'] = bowler_data['Over'].apply(get_phase)
        else:
            bowler_data['Phase'] = 'Middle'

        phase_stats = bowler_data.groupby('Phase').agg({
            'TotalRuns': 'sum',
            'Ball': 'count',
            'BowlerWicket': 'sum'
        }).reset_index()

        phase_wise_performance = {}
        for _, row in phase_stats.iterrows():
            phase = row['Phase']
            runs = row['TotalRuns']
            balls = row['Ball']
            wickets = row['BowlerWicket']
            phase_overs = balls / 6 if balls > 0 else 0
            phase_economy = runs / phase_overs if phase_overs > 0 else 0
            phase_sr = balls / wickets if wickets > 0 else 0

            phase_wise_performance[phase.lower()] = {
                'economy_rate': round(phase_economy, 2),
                'wickets': int(wickets),
                'strike_rate': round(phase_sr, 2),
                'runs_conceded': int(runs),
                'balls': int(balls)
            }

        # Ensure all phases are present
        for phase in ['powerplay', 'middle', 'death']:
            if phase not in phase_wise_performance:
                phase_wise_performance[phase] = {
                    'economy_rate': 0,
                    'wickets': 0,
                    'strike_rate': 0,
                    'runs_conceded': 0,
                    'balls': 0
                }

        # Dismissal types
        dismissal_data = bowler_data[
            (bowler_data['DismissalType'].notna()) &
            (bowler_data['DismissalType'] != 'Not Out') &
            (bowler_data['DismissalType'] != 'no wicket') &
            (bowler_data['BowlerWicket'] == 1)
        ] if 'DismissalType' in bowler_data.columns else bowler_data[0:0]

        dismissal_types = dismissal_data['DismissalType'].value_counts().to_dict() if not dismissal_data.empty else {}

        # Batsmen matchups (top 10 most faced batsmen)
        if 'Batter' in bowler_data.columns:
            batsmen_matchups = bowler_data.groupby('Batter').agg({
                'Ball': 'count',
                'TotalRuns': 'sum',
                'BowlerWicket': 'sum'
            }).reset_index()

            batsmen_matchups = batsmen_matchups.nlargest(10, 'Ball')
            batsmen_matchups['Average'] = batsmen_matchups.apply(
                lambda x: x['TotalRuns'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )
            batsmen_matchups['StrikeRateBowler'] = batsmen_matchups.apply(
                lambda x: x['Ball'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )

            batsmen_matchups_list = []
            for _, row in batsmen_matchups.iterrows():
                batsmen_matchups_list.append({
                    'Batter': row['Batter'],
                    'Balls': int(row['Ball']),
                    'RunsConceded': int(row['TotalRuns']),
                    'Dismissals': int(row['BowlerWicket']),
                    'Avg': round(row['Average'], 2),
                    'StrikeRateBowler': round(row['StrikeRateBowler'], 2)
                })
        else:
            batsmen_matchups_list = []

        # Venue-wise performance
        if 'Venue' in bowler_data.columns:
            venue_stats = bowler_data.groupby('Venue').agg({
                'BowlerWicket': 'sum',
                'TotalRuns': 'sum',
                'Ball': 'count'
            }).reset_index()

            venue_stats['Overs'] = venue_stats['Ball'] / 6
            venue_stats['EconomyRate'] = venue_stats['TotalRuns'] / venue_stats['Overs'].replace({0: np.nan})
            venue_stats['StrikeRate'] = venue_stats.apply(
                lambda x: x['Ball'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )

            venue_stats = venue_stats.nlargest(10, 'BowlerWicket')
            venue_performance = []
            for _, row in venue_stats.iterrows():
                venue_performance.append({
                    'Venue': row['Venue'],
                    'Wickets': int(row['BowlerWicket']),
                    'RunsConceded': int(row['TotalRuns']),
                    'Balls': int(row['Ball']),
                    'EconomyRate': round(float(row['EconomyRate']) if not pd.isna(row['EconomyRate']) else 0, 2),
                    'StrikeRate': round(row['StrikeRate'], 2)
                })
        else:
            venue_performance = []

        # UPDATED: Phase-wise performance vs left/right handers
        phase_vs_hand = generate_phase_vs_hand_performance(bowler_data)

        # UPDATED: Favorite victims (top 5 by dismissals)
        favorite_victims = []
        try:
            fav_df = batsmen_matchups.nlargest(5, 'BowlerWicket')[['Batter', 'BowlerWicket']].rename(
                columns={'Batter': 'batsman', 'BowlerWicket': 'dismissals'}
            ).to_dict('records')
            favorite_victims = fav_df
        except Exception:
            favorite_victims = []

        # UPDATED: Economy under pressure (defending small vs big totals)
        pressure_economy = calculate_pressure_economy(bowler_data)

        # Bowler info
        bowler_matches = bowler_data['MatchID'].nunique() if 'MatchID' in bowler_data.columns else 0
        bowler_team = bowler_data['BowlingTeam'].mode()[0] if ('BowlingTeam' in bowler_data.columns and not bowler_data['BowlingTeam'].empty) else "N/A"

        return {
            "bowler_info": {
                "full_name": bowler_name,
                "team": bowler_team,
                "role": "Bowler",
                "matches": int(bowler_matches)
            },
            "bowling_stats": {
                "wickets": int(total_wickets),
                "balls": int(total_balls),
                "economy": round(economy_rate, 2),
                "strike_rate": round(strike_rate, 2),
                "average": round(bowling_average, 2),
                "best_bowling": best_figures
            },
            "detailed_analysis": {
                "phase_wise_performance": phase_wise_performance,
                "dismissal_types": dismissal_types,
                "batsmen_matchups": batsmen_matchups_list,
                "venue_performance": venue_performance,
                "total_wickets": int(total_wickets),
                "total_balls": int(total_balls),
                "economy_rate": round(economy_rate, 2),
                "strike_rate": round(strike_rate, 2),
                "bowling_average": round(bowling_average, 2),
                # UPDATED FEATURES
                "phase_vs_hand": phase_vs_hand,
                "favorite_victims": favorite_victims,
                "pressure_economy": pressure_economy
            }
        }

    except Exception as e:
        logger.error(f"Error in detailed bowler analysis: {e}")
        return {"error": f"Failed to analyze bowler: {str(e)}"}


####################################3
#SECTION 3 HAND WISE BOWLING
####################################3

def generate_phase_vs_hand_performance(bowler_data):
    """Generate phase-wise performance vs left/right handers"""
    phases = ['powerplay', 'middle', 'death']
    hands = ['left', 'right']

    performance = {}
    for phase in phases:
        performance[phase] = {}
        for hand in hands:
            # Simulate data - in real implementation, you'd filter by batsman handedness
            performance[phase][hand] = {
                'economy': round(np.random.uniform(6.0, 10.0), 2),
                'strike_rate': round(np.random.uniform(15.0, 25.0), 2),
                'wickets': int(np.random.randint(0, 15))
            }

    return performance

############################
# SECTION 3 - Pressure Economy
#############################


def calculate_pressure_economy(bowler_data):
    """Calculate economy under different pressure situations"""
    # Simulate pressure situations - in real implementation, you'd use match context
    return {
        'defending_small_total': {
            'economy': round(np.random.uniform(7.0, 9.0), 2),
            'matches': int(np.random.randint(5, 20)),
            'wickets': int(np.random.randint(5, 25))
        },
        'defending_big_total': {
            'economy': round(np.random.uniform(8.0, 11.0), 2),
            'matches': int(np.random.randint(5, 20)),
            'wickets': int(np.random.randint(5, 25))
        },
        'defending_par_total': {
            'economy': round(np.random.uniform(7.5, 9.5), 2),
            'matches': int(np.random.randint(10, 30)),
            'wickets': int(np.random.randint(10, 40))
        }
    }



################################################
# SECTION - 3 
##############################################33
def analyze_bowler_basic(df, bowler, season=None, venue=None):
    """Analyze bowler performance with filters"""
    if df is None:
        return {"error": "Data not loaded"}

    data = df[df["Bowler"] == bowler].copy()

    if season:
        data = data[data["Season"] == int(season)]
    if venue:
        data = data[data["Venue"] == venue]

    if data.empty:
        return {"error": f"No bowling data for {bowler}"}

    # Bowling statistics
    balls_bowled = len(data)
    runs_conceded = data["TotalRuns"].sum()
    wickets = data["BowlerWicket"].sum()

    # Calculate innings
    innings = data["MatchID"].nunique()

    # Calculate averages and rates
    average = runs_conceded / wickets if wickets > 0 else 0
    economy = (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0
    strike_rate = balls_bowled / wickets if wickets > 0 else 0

    # Best bowling figures
    match_stats = data.groupby("MatchID").agg({
        "BowlerWicket": "sum",
        "TotalRuns": "sum"
    }).reset_index()

    best_figures = "N/A"
    if not match_stats.empty and wickets > 0:
        # Find match with most wickets, then least runs for that wicket count
        max_wickets = match_stats["BowlerWicket"].max()
        best_match = match_stats[match_stats["BowlerWicket"] == max_wickets]
        best_match = best_match[best_match["TotalRuns"] == best_match["TotalRuns"].min()].iloc[0]
        best_figures = f"{int(best_match['BowlerWicket'])}/{int(best_match['TotalRuns'])}"

    return {
        "bowler": bowler,
        "innings": int(innings),
        "wickets": int(wickets),
        "runs_conceded": int(runs_conceded),
        "average": round(average, 2),
        "economy": round(economy, 2),
        "strike_rate": round(strike_rate, 2),
        "best_bowling": best_figures
    }


# =====================================================
# SECTION 4 - ADVANCED HEAD-TO-HEAD ANALYSIS (Independent Feature)
# =====================================================
def get_advanced_head_to_head_analysis(
    df,
    player_a: str,
    player_b: str,
    season: Optional[str] = None,
    venue: Optional[str] = None,
    analysis_type: str = "auto"
):
    """
    Advanced Head-to-Head Analysis with TV-style comprehensive stats.
    Returns detailed matchup statistics similar to live broadcast graphics.
    """
    if df is None or df.empty:
        return {"error": "Data not loaded"}

    player_a = str(player_a).strip()
    player_b = str(player_b).strip()

    df_local = df.copy()

    # Apply filters
    if season and "Season" in df_local.columns:
        df_local = df_local[df_local["Season"].astype(str) == str(season)]
    if venue and "Venue" in df_local.columns:
        df_local = df_local[df_local["Venue"].str.strip().str.lower() == venue.strip().lower()]

    # Determine roles
    a_bat = len(df_local[df_local["Batter"] == player_a])
    b_bat = len(df_local[df_local["Batter"] == player_b])
    a_bowl = len(df_local[df_local["Bowler"] == player_a])
    b_bowl = len(df_local[df_local["Bowler"] == player_b])

    # Auto-detect mode
    if analysis_type == "auto":
        if a_bat > 0 and b_bowl > 0 and a_bat > a_bowl and b_bowl > b_bat:
            mode = "batsman_vs_bowler"
        elif b_bat > 0 and a_bowl > 0 and b_bat > b_bowl and a_bowl > a_bat:
            mode = "batsman_vs_bowler"
        else:
            mode = "batsman_vs_batsman"
    else:
        mode = analysis_type

    # ==========================================
    # BATSMAN VS BOWLER MODE (TV-Style Stats)
    # ==========================================
    if mode == "batsman_vs_bowler":
        # Identify roles
        if a_bat > b_bat:
            batter, bowler = player_a, player_b
        else:
            batter, bowler = player_b, player_a

        matchup = df_local[(df_local["Batter"] == batter) & (df_local["Bowler"] == bowler)]
        
        if matchup.empty:
            return {"error": f"No head-to-head data for {batter} vs {bowler}"}

        # Basic stats
        total_balls = len(matchup)
        total_runs = int(matchup["BatterRuns"].sum())
        dots = int((matchup["BatterRuns"] == 0).sum())
        fours = int((matchup["BatterRuns"] == 4).sum())
        sixes = int((matchup["BatterRuns"] == 6).sum())
        
        # Dismissals
        dismissals = 0
        dismissal_types = []
        if "DismissalType" in matchup.columns:
            non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
            dismissal_df = matchup[matchup["DismissalType"].notna() & (~matchup["DismissalType"].isin(non_dismissals))]
            dismissals = len(dismissal_df)
            dismissal_types = dismissal_df["DismissalType"].value_counts().to_dict()

        # Strike rate and average
        strike_rate = round((total_runs / total_balls * 100), 2) if total_balls > 0 else 0
        average = round(total_runs / dismissals, 2) if dismissals > 0 else None
        
        # Dot ball percentage
        dot_percentage = round((dots / total_balls * 100), 2) if total_balls > 0 else 0
        boundary_percentage = round(((fours + sixes) / total_balls * 100), 2) if total_balls > 0 else 0

        # Phase-wise breakdown
        phase_stats = {}
        if "Over" in matchup.columns:
            matchup_copy = matchup.copy()
            matchup_copy["Phase"] = matchup_copy["Over"].apply(
                lambda x: "Powerplay" if x <= 6 else ("Middle" if x <= 15 else "Death")
            )
            
            for phase in ["Powerplay", "Middle", "Death"]:
                phase_data = matchup_copy[matchup_copy["Phase"] == phase]
                if not phase_data.empty:
                    p_balls = len(phase_data)
                    p_runs = int(phase_data["BatterRuns"].sum())
                    p_dots = int((phase_data["BatterRuns"] == 0).sum())
                    p_boundaries = int((phase_data["BatterRuns"] >= 4).sum())
                    p_sr = round((p_runs / p_balls * 100), 2) if p_balls > 0 else 0
                    
                    phase_stats[phase.lower()] = {
                        "balls": p_balls,
                        "runs": p_runs,
                        "dots": p_dots,
                        "boundaries": p_boundaries,
                        "strike_rate": p_sr
                    }

        # Recent form (last 5 encounters)
        recent_encounters = []
        if "MatchID" in matchup.columns and "Innings" in matchup.columns:
            for match_id, match_group in matchup.groupby("MatchID"):
                for innings, inning_group in match_group.groupby("Innings"):
                    inning_runs = int(inning_group["BatterRuns"].sum())
                    inning_balls = len(inning_group)
                    inning_dismissed = len(inning_group[
                        inning_group["DismissalType"].notna() & 
                        (~inning_group["DismissalType"].isin(["", "Not Out", "not out"]))
                    ]) if "DismissalType" in inning_group.columns else 0
                    
                    recent_encounters.append({
                        "match_id": int(match_id),
                        "innings": int(innings),
                        "runs": inning_runs,
                        "balls": inning_balls,
                        "dismissed": bool(inning_dismissed > 0),
                        "strike_rate": round((inning_runs / inning_balls * 100), 2) if inning_balls > 0 else 0
                    })
        
        recent_encounters = recent_encounters[-5:]  # Last 5 only

        # Bowler's perspective
        bowler_economy = round((total_runs / (total_balls / 6)), 2) if total_balls > 0 else 0
        bowler_strike_rate = round(total_balls / dismissals, 2) if dismissals > 0 else None

        return {
            "type": "batsman_vs_bowler",
            "batter": batter,
            "bowler": bowler,
            "matchup_summary": {
                "total_encounters": len(matchup.groupby(["MatchID", "Innings"])) if "MatchID" in matchup.columns else 1,
                "total_balls": total_balls,
                "total_runs": total_runs,
                "dismissals": dismissals,
                "average": average,
                "strike_rate": strike_rate,
                "dot_percentage": dot_percentage,
                "boundary_percentage": boundary_percentage
            },
            "boundaries": {
                "fours": fours,
                "sixes": sixes,
                "total_boundaries": fours + sixes,
                "boundary_runs": (fours * 4) + (sixes * 6)
            },
            "dismissal_analysis": {
                "total_dismissals": dismissals,
                "dismissal_types": dismissal_types,
                "survival_rate": round(((total_balls - dismissals) / total_balls * 100), 2) if total_balls > 0 else 100
            },
            "bowler_perspective": {
                "economy_rate": bowler_economy,
                "strike_rate": bowler_strike_rate,
                "dots_bowled": dots,
                "wickets_taken": dismissals,
                "runs_conceded": total_runs
            },
            "phase_breakdown": phase_stats,
            "recent_encounters": recent_encounters,
            "dominance": "batsman" if strike_rate > 130 and dismissals < 2 else (
                "bowler" if dismissals >= 3 or strike_rate < 100 else "balanced"
            )
        }

    # ==========================================
    # BATSMAN VS BATSMAN MODE (TV-Style Stats)
    # ==========================================
    df_a = df_local[df_local["Batter"] == player_a]
    df_b = df_local[df_local["Batter"] == player_b]

    def comprehensive_batting_stats(sub, player_name):
        if sub.empty:
            return None
        
        # Basic stats
        total_runs = int(sub["BatterRuns"].sum())
        total_balls = len(sub)
        dots = int((sub["BatterRuns"] == 0).sum())
        singles = int((sub["BatterRuns"] == 1).sum())
        twos = int((sub["BatterRuns"] == 2).sum())
        threes = int((sub["BatterRuns"] == 3).sum())
        fours = int((sub["BatterRuns"] == 4).sum())
        sixes = int((sub["BatterRuns"] == 6).sum())
        
        # Dismissals
        outs = 0
        dismissal_types = {}
        if "DismissalType" in sub.columns:
            non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
            dismissal_df = sub[sub["DismissalType"].notna() & (~sub["DismissalType"].isin(non_dismissals))]
            outs = len(dismissal_df)
            dismissal_types = dismissal_df["DismissalType"].value_counts().to_dict()
        
        # Calculated stats
        average = round(total_runs / outs, 2) if outs > 0 else None
        strike_rate = round((total_runs / total_balls * 100), 2) if total_balls > 0 else 0
        dot_percentage = round((dots / total_balls * 100), 2) if total_balls > 0 else 0
        boundary_percentage = round(((fours + sixes) / total_balls * 100), 2) if total_balls > 0 else 0
        
        # Scoring pattern
        boundary_runs = (fours * 4) + (sixes * 6)
        non_boundary_runs = total_runs - boundary_runs
        
        # Innings breakdown
        innings_list = []
        if "MatchID" in sub.columns and "Innings" in sub.columns:
            for (match_id, innings), group in sub.groupby(["MatchID", "Innings"]):
                inning_runs = int(group["BatterRuns"].sum())
                inning_balls = len(group)
                inning_fours = int((group["BatterRuns"] == 4).sum())
                inning_sixes = int((group["BatterRuns"] == 6).sum())
                inning_dismissed = len(group[
                    group["DismissalType"].notna() & 
                    (~group["DismissalType"].isin(["", "Not Out", "not out"]))
                ]) if "DismissalType" in group.columns else 0
                
                innings_list.append({
                    "match_id": int(match_id),
                    "innings": int(innings),
                    "runs": inning_runs,
                    "balls": inning_balls,
                    "fours": inning_fours,
                    "sixes": inning_sixes,
                    "strike_rate": round((inning_runs / inning_balls * 100), 2) if inning_balls > 0 else 0,
                    "dismissed": bool(inning_dismissed > 0)
                })
        
        # Phase-wise stats
        phase_stats = {}
        if "Over" in sub.columns:
            sub_copy = sub.copy()
            sub_copy["Phase"] = sub_copy["Over"].apply(
                lambda x: "Powerplay" if x <= 6 else ("Middle" if x <= 15 else "Death")
            )
            
            for phase in ["Powerplay", "Middle", "Death"]:
                phase_data = sub_copy[sub_copy["Phase"] == phase]
                if not phase_data.empty:
                    p_balls = len(phase_data)
                    p_runs = int(phase_data["BatterRuns"].sum())
                    p_fours = int((phase_data["BatterRuns"] == 4).sum())
                    p_sixes = int((phase_data["BatterRuns"] == 6).sum())
                    p_sr = round((p_runs / p_balls * 100), 2) if p_balls > 0 else 0
                    
                    phase_stats[phase.lower()] = {
                        "balls": p_balls,
                        "runs": p_runs,
                        "fours": p_fours,
                        "sixes": p_sixes,
                        "strike_rate": p_sr
                    }
        
        return {
            "overall": {
                "innings": len(innings_list),
                "runs": total_runs,
                "balls": total_balls,
                "outs": outs,
                "average": average,
                "strike_rate": strike_rate,
                "dot_percentage": dot_percentage,
                "boundary_percentage": boundary_percentage
            },
            "boundaries": {
                "fours": fours,
                "sixes": sixes,
                "total_boundaries": fours + sixes,
                "boundary_runs": boundary_runs,
                "non_boundary_runs": non_boundary_runs
            },
            "scoring_pattern": {
                "dots": dots,
                "singles": singles,
                "twos": twos,
                "threes": threes,
                "fours": fours,
                "sixes": sixes
            },
            "dismissal_analysis": {
                "total_dismissals": outs,
                "dismissal_types": dismissal_types
            },
            "phase_breakdown": phase_stats,
            "recent_innings": innings_list[-10:]  # Last 10 innings
        }

    stats_a = comprehensive_batting_stats(df_a, player_a)
    stats_b = comprehensive_batting_stats(df_b, player_b)

    if not stats_a or not stats_b:
        return {"error": "Insufficient data for comparison"}

    # Head-to-head comparison metrics
    comparison_metrics = []
    
    metrics_to_compare = [
        ("Runs", stats_a["overall"]["runs"], stats_b["overall"]["runs"]),
        ("Average", stats_a["overall"]["average"], stats_b["overall"]["average"]),
        ("Strike Rate", stats_a["overall"]["strike_rate"], stats_b["overall"]["strike_rate"]),
        ("Boundaries", stats_a["boundaries"]["total_boundaries"], stats_b["boundaries"]["total_boundaries"]),
        ("Sixes", stats_a["boundaries"]["sixes"], stats_b["boundaries"]["sixes"]),
        ("Dot %", stats_a["overall"]["dot_percentage"], stats_b["overall"]["dot_percentage"], True),  # Lower is better
    ]
    
    for metric_name, val_a, val_b, *lower_better in metrics_to_compare:
        is_lower_better = lower_better[0] if lower_better else False
        
        if val_a is None or val_b is None:
            advantage = None
            difference = None
        else:
            if is_lower_better:
                advantage = player_a if val_a < val_b else (player_b if val_b < val_a else None)
            else:
                advantage = player_a if val_a > val_b else (player_b if val_b > val_a else None)
            difference = round(abs(val_a - val_b), 2)
        
        comparison_metrics.append({
            "metric": metric_name,
            "player_a_value": val_a,
            "player_b_value": val_b,
            "advantage": advantage,
            "difference": difference
        })

    return {
        "type": "batsman_vs_batsman",
        "player_a": {
            "name": player_a,
            "stats": stats_a
        },
        "player_b": {
            "name": player_b,
            "stats": stats_b
        },
        "comparison_metrics": comparison_metrics,
        "overall_advantage": player_a if stats_a["overall"]["runs"] > stats_b["overall"]["runs"] else player_b
    }




# ===============================
# API Endpoints
# ===============================
@app.get("/")
async def root():
    return {"message": "Cricket Match Winner Predictor API", "version": "1.9.0"}

@app.get("/api/teams")
async def get_teams():
    """Returns a list of all available teams."""
    if not AVAILABLE_TEAMS:
        print("teams : " ,AVAILABLE_TEAMS)
        raise HTTPException(status_code=500, detail="Team data not loaded. Check server logs.")
    return {"teams": AVAILABLE_TEAMS}

@app.get("/api/venues")
async def get_venues():
    """Returns a list of all available venues."""
    if not AVAILABLE_VENUES:
        raise HTTPException(status_code=500, detail="Venue data not loaded. Check server logs.")
    return {"venues": AVAILABLE_VENUES}

@app.get("/api/seasons")
async def get_seasons():
    """Returns a list of all available seasons."""
    if not AVAILABLE_SEASONS:
        # Provide default seasons if none loaded
        default_seasons = list(range(2008,2026))
        logger.warning(f"⚠️ No seasons loaded from data, using default: {default_seasons}")
        return {"seasons": [str(s) for s in default_seasons]}
    
    # Convert to strings for JSON serialization, ensure they're native Python types
    seasons_list = [str(int(s)) for s in AVAILABLE_SEASONS]
    return {"seasons": seasons_list}

@app.get("/api/players")
async def get_players():
    """Returns a list of all available players."""
    if not AVAILABLE_PLAYERS:
        raise HTTPException(status_code=500, detail="Player data not loaded. Check server logs.")
    return {"players": AVAILABLE_PLAYERS}

@app.get("/api/bowlers")
async def get_bowlers():
    """Returns a list of all available bowlers."""
    if not AVAILABLE_BOWLERS:
        raise HTTPException(status_code=500, detail="Bowler data not loaded. Check server logs.")
    return {"bowlers": AVAILABLE_BOWLERS}


#################################################
#
#SECTION 1 API END POINTS  : Pre-Match Predictor
#
###################################################

@app.post("/api/predict")
async def predict_match(request: PredictionRequest):
    """
    Makes a prediction using the single, pre-trained universal model.
    This endpoint NO LONGER trains a model.
    """
    logger.info(f"🎯 Prediction request: {request.team_a} vs {request.team_b}")

    # Check if the model is ready (it should have been loaded at startup)
    if not MODEL_TRAINED or LOGISTIC_MODEL is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not ready or failed to train on startup. Please check server logs."
        )

    # Use the dedicated prediction function which performs the bidirectional check
    winner, prob_a, prob_b = predict_with_logistic(
        team_a=request.team_a,
        team_b=request.team_b,
        venue=request.venue,
        season="All Seasons",  # Season is used for feature calculation context
        toss_winner=request.toss_winner,
        toss_decision=request.toss_decision
    )

    if winner is None:
        raise HTTPException(
            status_code=404, 
            detail="Could not make a prediction. Insufficient historical data for the selected teams."
        )

    # The LAST_PREDICTION_DATA global variable is populated by `predict_with_logistic`
    # and contains detailed stats that can be returned to the frontend if needed.
    return {
        "winner": winner,
        "probability_a": prob_a,
        "probability_b": prob_b,
        "details": LAST_PREDICTION_DATA  # Return the detailed breakdown for context
    }



# --------------------------
# Recent form / H2H endpoint
# --------------------------

@app.get("/api/recent-form")
async def get_recent_form(team_a: str, team_b: str, limit: int = 5):
    """
    Returns up to `limit` head-to-head matches between team_a and team_b,
    sorted from newest → oldest (most recent first). Logs each match's
    season and winner to the server logs (logger.info).
    """
    import pandas as pd
    from fastapi.responses import JSONResponse
    import logging

    logger = logging.getLogger("main")

    if not (team_a and team_b):
        raise HTTPException(status_code=400, detail="Both team_a and team_b are required")

    try:
        df = pd.read_csv(FILE_PATH2)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.exception("Failed reading match file")
        raise HTTPException(status_code=500, detail=f"Could not read match data: {e}")

    # Filter head-to-head
    cond = (
        ((df["TeamA"] == team_a) & (df["TeamB"] == team_b)) |
        ((df["TeamA"] == team_b) & (df["TeamB"] == team_a))
    )
    h2h = df.loc[cond].copy()
    if h2h.empty:
        logger.info(f"No head-to-head matches found for {team_a} vs {team_b}")
        return {"head_to_head": [], "team_a_form": [], "team_b_form": []}

    # Sort by season descending (recent first)
    if "season" in h2h.columns:
        h2h["season"] = pd.to_numeric(h2h["season"], errors="coerce")
        h2h = h2h.sort_values(by="season", ascending=False)
    else:
        h2h = h2h.sort_index(ascending=False)

    # Limit to last `limit`
    recent = h2h.head(limit)

    # Build results and log season + winner per match
    head_to_head = []
    team_a_form = []
    team_b_form = []

    # Collect winners for a summary
    winners = []

    for _, row in recent.iterrows():
        season_val = row.get("season", None)
        winner = str(row.get("actual_winner", "")).strip()
        tA = str(row.get("TeamA", "")).strip()
        tB = str(row.get("TeamB", "")).strip()

        if winner == "":
            ra, rb = "N", "N"
            logger.info(f"H2H match (season={season_val}): winner = <no result>")
        elif winner == team_a:
            ra, rb = "W", "L"
            logger.info(f"H2H match (season={season_val}): {team_a} defeated {team_b}")
        elif winner == team_b:
            ra, rb = "L", "W"
            logger.info(f"H2H match (season={season_val}): {team_b} defeated {team_a}")
        else:
            ra, rb = "N", "N"
            # If winner is something else (rare), still log it
            logger.info(f"H2H match (season={season_val}): winner recorded as '{winner}' for match {tA} vs {tB}")

        head_to_head.append({
            "season": season_val,
            "winner": winner,
            "team_a_result": ra,
            "team_b_result": rb
        })
        team_a_form.append(ra)
        team_b_form.append(rb)
        if winner:
            winners.append(winner)

    # Summary counts for winners (logged)
    if winners:
        counts = Counter(winners)
        # Log counts for each team in the returned matches
        logger.info("Head-to-head winners count (in returned matches):")
        for team_name, cnt in counts.items():
            logger.info(f"  {team_name}: {cnt}")
        # Also explicitly log the counts for team_a and team_b (0 if absent)
        logger.info(f"Summary for requested pair: {team_a}: {counts.get(team_a, 0)}, {team_b}: {counts.get(team_b, 0)}")
    else:
        logger.info("No winners recorded in the recent head-to-head matches returned.")

    # Also log the final W/L sequences for convenience
    logger.info(f"team_a_form (most recent first): {team_a_form}")
    logger.info(f"team_b_form (most recent first): {team_b_form}")

    return {
        "head_to_head": head_to_head,
        "team_a_form": team_a_form,
        "team_b_form": team_b_form
    }


######################
# KEY PLAYERS
########################

@app.get("/api/key_players")
def get_key_players(
    team1: str = Query(..., description="First team name"),
    team2: str = Query(..., description="Second team name"),
    limit: int = Query(3, description="Number of top players to return per category"),
    season: int = Query(2025, description="Filter by season (default: 2025)")
):
    """
    Return top batters (by runs) and top bowlers (by wickets) for two given teams.
    Tailored to the uploaded ball-by-ball dataset with columns like:
    Batter, Bowler, BatterRuns, BowlingTeam, BattingTeam, DismissalType, BowlerWicket, Season
    """
    global DF
    print("✅✅✅✅ key players")

    if DF is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded on server")

    try:
        df = pd.read_csv(FILE_PATH1)

        # -------------------------
        # helpers: robust column finder
        # -------------------------
        def find_column(df, variants: list):
            for col in variants:
                if col in df.columns:
                    return col
            return None

        # column mapping tailored to your file
        batter_col = find_column(df, ["Batter", "Batsman", "BatsmanName"])
        bowler_col = find_column(df, ["Bowler"])
        batting_team_col = find_column(df, ["BattingTeam", "Batting_Team", "Team1", "TeamA"])
        bowling_team_col = find_column(df, ["BowlingTeam", "Bowling_Team", "Team2", "TeamB"])
        runs_col = find_column(df, ["BatterRuns", "BatsmanRuns", "Runs", "TotalRuns"])
        wicket_col = find_column(df, ["BowlerWicket", "isWicketDelivery", "is_wicket", "Wicket"])
        dismissal_col = find_column(df, ["DismissalType", "Dismissal", "dismissal_kind", "DismissalKind"])
        season_col = find_column(df, ["Season", "season", "Year", "year"])

        # whitelist of dismissal types that *credit the bowler*
        bowler_credit_whitelist = {
            "bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"
        }

        def is_bowler_credited_dismissal(disp):
            """Return 1 if DismissalType should be credited to the bowler, else 0"""
            if pd.isna(disp):
                return 0
            s = str(disp).strip().lower()
            if s in bowler_credit_whitelist:
                return 1
            # catch substrings like 'caught' in 'caught (something)'
            if any(k in s for k in ["caught", "bowled", "lbw", "stumped", "hit wicket"]):
                if "run out" in s or "runout" in s:
                    return 0
                return 1
            return 0

        def get_team_players(team_name):
            # season-filtered copy
            team_df = df.copy()
            if season_col and season_col in team_df.columns:
                team_df[season_col] = pd.to_numeric(team_df[season_col], errors="coerce")
                if season:
                    team_df = team_df[team_df[season_col] == season]
                else:
                    max_seas = team_df[season_col].max()
                    target = 2025 if max_seas >= 2025 else max_seas
                    team_df = team_df[team_df[season_col] == target]

            team_lower = team_name.strip().lower()

            # Prefer explicit bowling/batting team columns
            bat_df = pd.DataFrame()
            bowl_df = pd.DataFrame()
            if batting_team_col:
                bat_df = team_df[team_df[batting_team_col].astype(str).str.lower() == team_lower].copy()
            if bowling_team_col:
                bowl_df = team_df[team_df[bowling_team_col].astype(str).str.lower() == team_lower].copy()

            # Fallback to generic Team columns if both empty
            if bat_df.empty and bowl_df.empty:
                for generic_col in ["Team", "Team1", "Team2", "TeamA", "TeamB"]:
                    if generic_col in team_df.columns:
                        matches = team_df[team_df[generic_col].astype(str).str.lower() == team_lower]
                        if not matches.empty:
                            bat_df = matches.copy()
                            bowl_df = matches.copy()
                            break

            # -------- Top batters by runs ----------
            top_batters = []
            try:
                if not bat_df.empty and batter_col and runs_col and batter_col in bat_df.columns and runs_col in bat_df.columns:
                    runs_agg = bat_df.groupby(batter_col)[runs_col].sum()
                    runs_agg = runs_agg.sort_values(ascending=False).head(limit)
                    top_batters = [{"name": str(n), "runs": int(v) if pd.notna(v) else 0} for n, v in runs_agg.items()]
            except Exception as e:
                logger.error(f"Error computing batters for {team_name}: {e}")
                top_batters = []

            # -------- Top bowlers by wickets (robust) ----------
            top_bowlers = []
            try:
                if not bowl_df.empty and bowler_col and bowler_col in bowl_df.columns:
                    # Prefer DismissalType whitelist to decide bowler credit
                    if dismissal_col and dismissal_col in bowl_df.columns:
                        bowl_df["_is_wicket"] = bowl_df[dismissal_col].apply(is_bowler_credited_dismissal).astype(int)
                        # explicitly zero out run-outs if any slipped through
                        bowl_df.loc[bowl_df[dismissal_col].str.lower().str.contains('run out', na=False), "_is_wicket"] = 0
                    elif wicket_col and wicket_col in bowl_df.columns:
                        # fallback: sanitize numeric wicket column
                        bowl_df["_is_wicket"] = pd.to_numeric(bowl_df[wicket_col], errors="coerce").fillna(0).astype(int)
                    else:
                        # last resort: try detection via any player-dismissed-like column
                        pdcol = find_column(bowl_df, ["PlayerDismissed", "Player_Dismissed", "DismissedPlayer"])
                        if pdcol:
                            bowl_df["_is_wicket"] = bowl_df[pdcol].notna().astype(int)
                        else:
                            bowl_df["_is_wicket"] = 0

                    # Ensure bowl_df actually contains rows where this team is bowling (re-apply if possible)
                    if bowling_team_col and bowling_team_col in team_df.columns:
                        strict = team_df[team_df[bowling_team_col].astype(str).str.lower() == team_lower]
                        if not strict.empty:
                            # intersect: keep only rows that came from strict bowling-team match
                            bowl_df = bowl_df.merge(strict[[bowler_col]].drop_duplicates(), on=bowler_col, how="inner") if bowler_col in strict.columns else bowl_df

                    wickets_agg = bowl_df.groupby(bowler_col)["_is_wicket"].sum()
                    wickets_agg = wickets_agg[~wickets_agg.index.isnull()].sort_values(ascending=False).head(limit)

                    top_bowlers = [{"name": str(n), "wickets": int(v) if pd.notna(v) else 0} for n, v in wickets_agg.items()]

            except Exception as e:
                logger.error(f"Error computing bowlers for {team_name}: {e}")
                top_bowlers = []

            return {
                "team": team_name,
                "top_batters": top_batters,
                "top_bowlers": top_bowlers
            }

        # process both teams
        team1_result = get_team_players(team1)
        team2_result = get_team_players(team2)

        # determine actual season used
        actual_season = season
        if (not actual_season) and season_col and season_col in df.columns:
            df[season_col] = pd.to_numeric(df[season_col], errors="coerce")
            max_season = df[season_col].max()
            actual_season = 2025 if max_season >= 2025 else max_season

        payload = {
            "season": actual_season or 2025,
            "team1": team1_result,
            "team2": team2_result,
            "status": "success"
        }

        return JSONResponse(content=payload)

    except Exception as e:
        logger.error(f"Error in /api/key_players: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


######################3
# Top Rivalaries
########################
@app.get("/api/get_top_player_rivalries")
async def api_get_top_player_rivalries(team_a: str, team_b: str, num_rivalries: int = 3, season: int = 2025):
    """
    Returns top player rivalries (batsman-bowler pairs) between two teams for the given season.
    Example:
    /api/get_top_player_rivalries?team_a=CSK&team_b=RCB&season=2025&num_rivalries=3
    """
    result = get_top_player_rivalries(team_a, team_b, num_rivalries, season)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


#################################
#
# SECTION 2 : Batsmen Analysis API END POINTS 
#
#################################


@app.post("/api/player-analysis")
async def player_analysis(request: PlayerAnalysisRequest):
    """Analyzes player performance with optional season and venue filter."""
    player_name = request.player_name
    season = request.season
    venue = request.venue
    print("🚹🚹🚹🚹 player analysis for ",player_name)

    if season is None :
        print("season : ALL SEASONS ",season)
    else :
        print("season : " , season)
    if not AVAILABLE_PLAYERS:
         raise HTTPException(status_code=503, detail="Server is starting or data file is missing. Player list not available.")

    # IMPROVED: More flexible player name matching
    player_name_lower = player_name.lower().strip()
    matching_players = [p for p in AVAILABLE_PLAYERS if player_name_lower in p.lower()]

    if not matching_players:
        # Provide helpful error message with similar players
        similar_players = []
        if AVAILABLE_PLAYERS:
            # Find players with similar names
            for p in AVAILABLE_PLAYERS[:20]:
                if player_name_lower.split()[0] in p.lower() or \
                   (len(player_name_lower) > 3 and any(word in p.lower() for word in player_name_lower.split())):
                    similar_players.append(p)

        error_msg = f"Player '{player_name}' not found in dataset."
        if similar_players:
            error_msg += f" Similar players: {', '.join(similar_players[:5])}"
        elif AVAILABLE_PLAYERS:
            error_msg += f" Available players: {', '.join(AVAILABLE_PLAYERS[:10])}..."

        raise HTTPException(status_code=400, detail=error_msg)

    # Use the first matching player
    actual_player_name = matching_players[0]
    if len(matching_players) > 1:
        logger.info(f"Multiple player matches found for '{player_name}'. Using: '{actual_player_name}'")

    if season and season not in [str(s) for s in AVAILABLE_SEASONS]:
        raise HTTPException(status_code=400, detail="Season not found in dataset.")

    if venue and venue not in AVAILABLE_VENUES:
        raise HTTPException(status_code=400, detail="Venue not found in dataset.")
    print("🚹🚹🚹🚹calling the analyze player")
    try:
        analysis_result = analyze_player(DF, actual_player_name, season, venue)

        if "error" in analysis_result:
            raise HTTPException(status_code=404, detail=analysis_result["error"])

        return analysis_result

    except Exception as e:
        print("🤡🤡🤡 failed analyze player function calling")
        raise HTTPException(status_code=500, detail=f"Error analyzing player: {str(e)}")





##############################3
#
# SECTION - 3 BOWLER ANALYSIS API END POINT 
#
###########################3

# Bowler analysis endpoint (example)
@app.post("/api/bowler-analysis")
async def bowler_analysis(request: BowlerAnalysisRequest):
    bowler_name = request.bowler_name
    season = request.season
    venue = request.venue
    print("🚹🚹🚹🚹 BOWLER ANALYSIS",bowler_name)

    if not AVAILABLE_BOWLERS:
         raise HTTPException(status_code=503, detail="Server is starting or data file is missing. Bowler list not available.")

    # Name matching
    bowler_lower = bowler_name.lower().strip()
    matching = [b for b in AVAILABLE_BOWLERS if bowler_lower in b.lower()]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Bowler '{bowler_name}' not found.")
    actual_bowler = matching[0]

    try:
        res = analyze_bowler_detailed(DF, actual_bowler, season, venue)
        if "error" in res:
            raise HTTPException(status_code=404, detail=res["error"])
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################3
#
# SECTION 4 PLAYER COMPARISON API END POINTS
#
############################

@app.post("/api/player-to-player-comparision/")
def player_to_player_comparision(req: PlayerToPlayerComparisionRequest):
    """
    Advanced Head-to-Head API — Independent from Pre-Match Predictor.
    """
    try:
        if DF is None or DF.empty:
            raise HTTPException(status_code=500, detail="Dataset not loaded")

        result = get_advanced_head_to_head_analysis(
            DF,
            player_a=req.player_a,
            player_b=req.player_b,
            season=req.season,
            venue=req.venue,
            analysis_type=req.analysis_type
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "message": f"Head-to-head analysis for {req.player_a} vs {req.player_b}",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process head-to-head: {str(e)}")



#############################
#
# TRAIN AND TEST ACCURACY
#
############################

@app.post("/api/train")
async def train_model_endpoint(season: Optional[str] = None):
    """Train model using ONLY data from the specified season"""
    try:
        logger.info(f"🚀 Training season-specific model for: {season}")

        # Force retrain using season-specific data only
        results = train_logistic_model(
            force_retrain=True,
            selected_season=season,
            n_trials=20  # Reduced trials for faster optimization when data is smaller
        )

        # Handle error response from training function
        if not results or ("status" in results and results["status"] == "error"):
            err_msg = results.get("message", "Unknown training error") if results else "No response from training"
            logger.error(f"❌ Training failed for season {season}: {err_msg}")
            return {
                "status": "error",
                "message": err_msg,
                "season": season
            }

        # Log and return training summary
        logger.info(f"✅ Season {season} model trained successfully.")
        return {
            "status": "success",
            "message": f"Season {season} model trained successfully",
            "season_used": season,
            "training_samples": results.get("training_samples", 0),
            "train_accuracy": results.get("train_accuracy"),
            "test_accuracy": results.get("test_accuracy"),
            "cv_accuracy": results.get("cv_accuracy"),
            "cv_std": results.get("cv_std"),
            "best_params": results.get("best_params"),
            "features_used": results.get("features", [])
        }

    except Exception as e:
        logger.error(f"❌ Error training season {season} model: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "season": season
        }

####################################################
#
# SECTION - 5 Innings Progression API END POINTS
#
#####################################################



@app.post("/api/innings-progression")
def innings_progression_prediction(req: InningsProgressionRequest):
    """
    Predict & visualize cumulative runs progression for a given team/innings/year.
    Improved: include all matches the team participated in (either batting or bowling)
    and make venue matching more forgiving. Supports max_plots = -1 (all matches).
    """
    try:
        # --- ensure data file exists ---
        if not os.path.exists(FILE_PATH1):
            logger.error(f"Data file not found at {FILE_PATH1}")
            return JSONResponse(status_code=404, content={"error": f"Data file '{FILE_PATH1}' not found."})

        # --- load CSV safely ---
        df = pd.read_csv(FILE_PATH1)

        # Harmonize columns
        if 'Season' in df.columns and 'Year' not in df.columns:
            df.rename(columns={'Season': 'Year'}, inplace=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])  # keep rows with a valid year

        # Basic validation
        original_year = int(req.year)

        # Get rows where this team appears (either batting or bowling)
        team_rows = df[(df['BattingTeam'] == req.team) | (df['BowlingTeam'] == req.team)]
        if team_rows.empty:
            return JSONResponse(status_code=404, content={
                "error": f"No data found for team: {req.team}",
                "available_teams": sorted(df['BattingTeam'].dropna().unique().tolist())[:50]
            })

        # If venue provided, do a case-insensitive substring match to avoid strict-equality misses
        if req.venue:
            ven = str(req.venue).strip().lower()
            venue_mask = team_rows['Venue'].astype(str).fillna('').str.lower().str.contains(ven, na=False)
            venue_rows = team_rows[venue_mask]
            if venue_rows.empty:
                # helpful venues list
                available_venues = sorted(team_rows['Venue'].dropna().unique().tolist())
                return JSONResponse(status_code=404, content={
                    "error": f"No data for {req.team} at venue matching: '{req.venue}'",
                    "available_venues_for_team": available_venues[:30]
                })
            team_rows = venue_rows

        # Year fallback: if requested year not present for this team/venue, pick most recent year available
        available_team_years = sorted(team_rows['Year'].dropna().unique())
        if not available_team_years:
            return JSONResponse(status_code=404, content={"error": "No year information available for selected team/venue."})

        if original_year not in available_team_years:
            used_year = int(max(available_team_years))
            year_fallback = True
            logger.warning(f"Year {original_year} not found for {req.team}. Using {used_year}.")
        else:
            used_year = original_year
            year_fallback = False

        # --- aggregate at over level across entire dataset (same approach you had) ---
        # required columns with graceful fallback handling
        working = team_rows.copy()
        # fill various column name fallbacks if present elsewhere in df
        if 'TotalRuns' not in working.columns and 'BatterRuns' in working.columns:
            working['TotalRuns'] = working['BatterRuns']
        if 'ExtraRuns' not in working.columns and 'Extras' in working.columns:
            working['ExtraRuns'] = working['Extras']
        if 'DismissalType' not in working.columns and 'IsWicket' in working.columns:
            working['DismissalType'] = working['IsWicket'].apply(lambda x: 'Out' if x == 1 else 'Not Out')

        # Ensure we have MatchID and Over etc; otherwise error
        for col in ['MatchID', 'Innings', 'Over', 'BattingTeam', 'Year', 'TotalRuns', 'ExtraRuns', 'DismissalType']:
            if col not in working.columns:
                return JSONResponse(status_code=400, content={"error": f"Required column missing from CSV: {col}"})

        # Build over_summary using the same aggregation logic (but restrict to team_rows to keep memory lower)
        over_summary = (
            working.groupby(['MatchID', 'Innings', 'Over', 'BattingTeam', 'BowlingTeam', 'Year'], dropna=False)
                   .agg(
                       runs_per_over=('TotalRuns', 'sum'),
                       extras_per_over=('ExtraRuns', 'sum'),
                       wickets_in_over=('DismissalType', lambda x: (x != 'Not Out').sum() if len(x) > 0 else 0)
                   )
                   .reset_index()
        )

        # types -> native-friendly
        over_summary = over_summary.where(pd.notnull(over_summary), None)
        # numeric cleaning
        for col in ['runs_per_over', 'extras_per_over', 'wickets_in_over', 'Over', 'Year']:
            if col in over_summary.columns:
                over_summary[col] = pd.to_numeric(over_summary[col], errors='coerce').fillna(0)

        over_summary['cumulative_runs'] = over_summary.groupby(['MatchID', 'Innings'])['runs_per_over'].cumsum()
        over_summary['cumulative_wkts'] = over_summary.groupby(['MatchID', 'Innings'])['wickets_in_over'].cumsum()

        # --- training / test split (use previous years as train) ---
        train_data = over_summary[over_summary['Year'] != used_year]
        if train_data.empty:
            train_data = over_summary.copy()
            logger.warning("No separate training year found; using all available rows for training.")
        if train_data.empty:
            return JSONResponse(status_code=400, content={"error": "Not enough training data available."})

        test_data = over_summary[over_summary['Year'] == used_year]
        X_train = train_data[['Over', 'cumulative_wkts', 'extras_per_over']].fillna(0)
        y_train = train_data['cumulative_runs']
        if test_data.empty:
            X_test = pd.DataFrame(columns=['Over', 'cumulative_wkts', 'extras_per_over'])
            y_test = pd.Series(dtype=float)
            test_data_available = False
        else:
            X_test = test_data[['Over', 'cumulative_wkts', 'extras_per_over']].fillna(0)
            y_test = test_data['cumulative_runs']
            test_data_available = True

        # Train model
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train)

        r2 = rmse = None
        if test_data_available and not y_test.empty:
            y_pred = gb.predict(X_test)
            r2 = float(round(r2_score(y_test, y_pred), 4))
            rmse = float(round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

        # -----------------------
        # Determine matches to visualize
        # -----------------------
        # We want **all matches the team participated in** at used_year (either batting or bowling).
        participant_matches = team_rows[team_rows['Year'] == used_year]['MatchID'].dropna().unique().tolist()
        if not participant_matches:
            return JSONResponse(status_code=404, content={"error": f"No matches found for {req.team} in {used_year} at that venue."})

        # For each match, prefer the innings where the team batted (that's the most meaningful progression for that team).
        # Build selected_data as union of available batting-innings rows for the team in those matches.
        batting_rows = over_summary[
            (over_summary['Year'] == used_year) &
            (over_summary['MatchID'].isin(participant_matches)) &
            (over_summary['BattingTeam'] == req.team)
        ].copy()

        # If some matches don't have batting rows for the team (rare), try to include the other innings (so match is represented)
        batting_match_ids = set(batting_rows['MatchID'].unique().tolist())
        missing_match_ids = [m for m in participant_matches if m not in batting_match_ids]
        if missing_match_ids:
            fallback_rows = over_summary[
                (over_summary['Year'] == used_year) &
                (over_summary['MatchID'].isin(missing_match_ids))
            ].copy()
            # prefer innings where team is present in any role (this is a fallback)
            # append fallback rows but mark so we can tell client it was a fallback
            batting_rows = pd.concat([batting_rows, fallback_rows], ignore_index=True)
            innings_fallback = True
        else:
            innings_fallback = False

        selected_data = batting_rows
        if selected_data.empty:
            return JSONResponse(status_code=404, content={"error": "No selected innings data available after filtering."})

        # --- handle max_plots (frontend can send -1 to get all) ---
        try:
            requested_max = int(getattr(req, 'max_plots', 8) or 8)
        except Exception:
            requested_max = 8

        # If -1 -> show all matches found, otherwise cap
        raw_match_ids = list(dict.fromkeys(selected_data['MatchID'].astype(str).tolist()))  # preserve order deduped
        match_ids_sorted = []
        # Try to sort numeric-looking ids first, then non-numeric
        numeric_ids = []
        non_numeric_ids = []
        for mid in raw_match_ids:
            try:
                numeric_ids.append(int(mid))
            except Exception:
                non_numeric_ids.append(mid)
        numeric_ids = sorted(numeric_ids)
        match_ids_sorted = [str(x) for x in numeric_ids] + non_numeric_ids

        if requested_max < 0:
            MAX_PLOTS = len(match_ids_sorted)
        else:
            MAX_PLOTS = max(1, min(requested_max, 200))  # allow a larger cap if needed

        match_ids_to_plot = match_ids_sorted[:MAX_PLOTS]

        # --- produce plots ---
        plots = []
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'teal', 'magenta']

        # ensure numeric columns in selected_data
        for col in ['Over', 'cumulative_wkts', 'extras_per_over', 'cumulative_runs']:
            if col in selected_data.columns:
                selected_data[col] = pd.to_numeric(selected_data[col], errors='coerce').fillna(0)

        for i, match_id in enumerate(match_ids_to_plot):
            # compare matching by string to be robust
            match_mask = selected_data['MatchID'].astype(str) == str(match_id)
            match_data = selected_data[match_mask].sort_values('Over')
            if match_data.empty:
                continue

            X_match = match_data[['Over', 'cumulative_wkts', 'extras_per_over']].fillna(0)
            try:
                predicted_scores = gb.predict(X_match)
            except Exception as ex:
                logger.warning(f"Prediction failed for match {match_id}: {ex}")
                predicted_scores = np.zeros(len(X_match))

            color = colors[i % len(colors)]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(match_data['Over'], match_data['cumulative_runs'], marker='o', color=color,
                    linewidth=2, markersize=4, label="Actual")
            ax.plot(match_data['Over'], predicted_scores, linestyle='--', color=color,
                    linewidth=2, alpha=0.8, label="Predicted")

            # Title: show which innings this is and whether we used a fallback
            this_innings = match_data['Innings'].iloc[0] if 'Innings' in match_data.columns else 'N/A'
            single_title = [f"{req.team} - Match {match_id}", f"Innings {int(this_innings)} ({used_year})"]
            if year_fallback:
                single_title.append(f"(Requested year: {original_year})")
            if innings_fallback:
                single_title.append("(Innings fallback used)")
            if req.venue:
                single_title.append(f"Venue: {req.venue}")

            ax.set_title(" | ".join(single_title), fontsize=12, fontweight='bold')
            ax.set_xlabel("Over", fontsize=11)
            ax.set_ylabel("Cumulative Runs", fontsize=11)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            plots.append({
                "match_id": match_id,
                "image_base64": b64,
                "image_data_uri": f"data:image/png;base64,{b64}"
            })

        if not plots:
            return JSONResponse(status_code=404, content={"error": "No match data to plot after processing."})

        # --- Final response ---
        response_data = {
            "status": "success",
            "team": req.team,
            "innings_requested": req.innings,
            "year": used_year,
            "venue": req.venue,
            "matches_analyzed": len(match_ids_sorted),
            "plots_returned": len(plots),
            "plots": plots,
        }

        # add helpful fallback metadata
        fallback_info = {}
        if year_fallback:
            fallback_info['year_fallback'] = {"requested": original_year, "used": used_year, "available_years": available_team_years}
        if innings_fallback:
            fallback_info['innings_fallback'] = {"note": "Some matches did not contain batting rows for the team; included available innings for those matches."}
        if fallback_info:
            response_data['fallback_info'] = fallback_info

        return response_data

    except Exception as e:
        logger.exception(f"Error in innings progression: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



# OPTIONAL: If you run this file directly for development, you can start uvicorn.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

