import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BakuApartmentDataset.xlsx")
NUM_COLS = ["rooms", "area", "floor", "building_floors", "new_building", "repair", "room_area"]


@st.cache_data
def load_data():
    """All ads, parsed into clean numeric columns (nothing deleted yet)."""
    df = pd.read_excel(DATA_FILE)
    out = pd.DataFrame()
    out["price"] = df["Qiymət"].str.replace(r"[^\d]", "", regex=True).astype(float)
    out["area"] = df["Sahə"].str.replace("m²", "").str.replace(",", ".").str.strip().astype(float)
    floors = df["Mərtəbə"].str.split("/", expand=True).astype(float)
    out["floor"], out["building_floors"] = floors[0], floors[1]
    out["rooms"] = df["Otaq Sayı"]
    out["new_building"] = (df["Yeni Tikili"] == "Yeni tikili").astype(int)
    out["repair"] = (df["Təmir"] == "var").astype(int)
    out["bill_of_sale"] = (df["Çıxarış"] == "var").astype(int)
    out["mortgage"] = (df["İpoteka"] == "var").astype(int)
    out["location"] = df["Ərazi"]
    out["room_area"] = out["area"] / out["rooms"]
    out["price_m2"] = out["price"] / out["area"]
    return out


@st.cache_data
def load_clean():
    """Same data without obvious typing mistakes (e.g. a 345,000 m² flat, floor above the roof)."""
    d = load_data()
    d = d[d["area"].between(12, 1600) & (d["floor"] <= d["building_floors"])]
    return d.reset_index(drop=True)


RARE_LIMIT = 50  # places with fewer ads than this are grouped as "Others"


def location_type(loc):
    """Azerbaijani place names end with m. (metro), r. (district) or q. (settlement)."""
    for suffix, kind in ((" m.", "metro"), (" r.", "district"), (" q.", "settlement")):
        if str(loc).endswith(suffix):
            return kind
    return "other"


@st.cache_data
def rare_locations():
    counts = load_clean()["location"].value_counts()
    return set(counts[counts < RARE_LIMIT].index)


def make_features(d, columns=None):
    group = d["location"].where(~d["location"].isin(rare_locations()), "Others")
    kind = d["location"].map(location_type)
    X = pd.concat(
        [d[NUM_COLS], pd.get_dummies(group, prefix="loc"), pd.get_dummies(kind, prefix="type")],
        axis=1,
    )
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


@st.cache_resource(show_spinner="Training the model (first visit only, about 20 seconds)...")
def train_model():
    df = load_clean()
    X = make_features(df)
    cols = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, df["price"], test_size=0.2, random_state=42)
    test_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = test_model.predict(Xte)
    err = np.abs(pred - yte) / yte
    stats = {
        "r2": float(r2_score(yte, pred)),
        "median_err": float(np.median(err)),
        "within10": float((err <= 0.10).mean()),
        "within20": float((err <= 0.20).mean()),
        "n_test": int(len(yte)),
    }
    final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X, df["price"])
    return final, cols, stats
