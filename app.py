# app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import joblib

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ada_final_full.joblib"
FEATURE_COLS_PATH = BASE_DIR / "feature_columns.pkl"
FEATURES_LATEST_PATH = BASE_DIR / "features_latest.csv"
TICKER_NAMES_PATH = BASE_DIR / "ticker_names.csv"   # put your ticker/company CSV here

# ------------------------------------------------------------
# Flask app
# ------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ------------------------------------------------------------
# Load model + feature list once at startup
# ------------------------------------------------------------
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)   # list of feature names used for training

# ------------------------------------------------------------
# Helper: load latest features and compute predictions
# ------------------------------------------------------------
def load_and_score_latest():
    """
    Load features_latest.csv, join company names, compute predictions,
    and return a DataFrame with 'predicted_score' and 'company'.
    """
    df = pd.read_csv(FEATURES_LATEST_PATH)

    # ---- join company names (if mapping file exists) ----
    if TICKER_NAMES_PATH.exists():
        names = pd.read_csv(TICKER_NAMES_PATH)
        # expect columns: 'ticker', 'company'
        df = df.merge(names, on="ticker", how="left")
    else:
        df["company"] = np.nan

    # if company is missing, fall back to ticker
    df["company"] = df["company"].fillna(df["ticker"])

    # ---- feature checks & predictions ----
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns in features_latest.csv: {missing}"
        )

    X_latest = df[feature_cols].copy()

    if X_latest.isna().sum().sum() != 0:
        raise ValueError("features_latest.csv still contains NaNs in feature columns.")

    preds = model.predict(X_latest)
    df["predicted_score"] = preds

    return df


# Load once at startup
df_latest = load_and_score_latest()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/top-bottom", methods=["GET"])
def top_bottom():
    """
    Return top and bottom K stocks by predicted_score,
    with global rank (1 = best) and company name.
    """
    global df_latest

    # Optional refresh from CSV
    refresh = request.args.get("refresh", "0")
    if refresh == "1":
        df_latest = load_and_score_latest()

    # K handling
    try:
        k = int(request.args.get("k", 20))
    except ValueError:
        k = 20

    n = len(df_latest)
    k = max(1, min(k, n // 2))  # at least 1, at most half the universe

    # Rank descending by predicted_score  (1 = best)
    ranked = df_latest.sort_values("predicted_score", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # Top K and bottom K by rank
    top = ranked.head(k)
    bottom = ranked.tail(k).sort_values("rank", ascending=True)

    def format_block(block):
        return [
            {
                "ticker": row["ticker"],
                "company": row["company"],
                "rank": int(row["rank"]),
                # keep raw score in case you ever want it later
                "score": float(row["predicted_score"]),
            }
            for _, row in block.iterrows()
        ]

    response = jsonify({
        "k": int(k),
        "n_universe": int(n),
        "top": format_block(top),
        "bottom": format_block(bottom),
    })

    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(debug=True)
