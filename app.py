import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "BakuApartmentDataset.xlsx"  # keep this file next to app.py

st.set_page_config(page_title="Baku Apartment Price Predictor", page_icon="🏙️")


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)
    out = pd.DataFrame()
    out["price"] = df["Qiymət"].str.replace(r"[^\d]", "", regex=True).astype(float)
    out["area"] = (
        df["Sahə"].str.replace("m²", "").str.replace(",", ".").str.strip().astype(float)
    )
    floors = df["Mərtəbə"].str.split("/", expand=True).astype(float)
    out["floor"], out["building_floors"] = floors[0], floors[1]
    out["rooms"] = df["Otaq Sayı"]
    out["new_building"] = (df["Yeni Tikili"] == "Yeni tikili").astype(int)
    out["repair"] = (df["Təmir"] == "var").astype(int)
    out["location"] = df["Ərazi"]
    # drop obvious typing mistakes (e.g. a flat of 345,000 m²) and impossible floors
    out = out[(out["area"].between(12, 1600)) & (out["floor"] <= out["building_floors"])]
    out["room_area"] = out["area"] / out["rooms"]
    return out.reset_index(drop=True)


def make_features(d, columns):
    X = d[["rooms", "area", "floor", "building_floors", "new_building", "repair", "room_area"]].copy()
    X = pd.concat([X, pd.get_dummies(d["location"], prefix="loc")], axis=1)
    return X.reindex(columns=columns, fill_value=0)


@st.cache_resource
def train(df):
    X_all = pd.concat(
        [
            df[["rooms", "area", "floor", "building_floors", "new_building", "repair", "room_area"]],
            pd.get_dummies(df["location"], prefix="loc"),
        ],
        axis=1,
    )
    cols = list(X_all.columns)
    Xtr, Xte, ytr, yte = train_test_split(X_all, df["price"], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = model.predict(Xte)
    err = np.abs(pred - yte) / yte
    stats = {
        "r2": r2_score(yte, pred),
        "median_err": float(np.median(err)),
        "within10": float((err <= 0.10).mean()),
        "within20": float((err <= 0.20).mean()),
        "n_test": len(yte),
    }
    final = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_all, df["price"])
    return final, cols, stats


df = load_data()
model, columns, stats = train(df)

st.title("🏙️ Baku Apartment Price Predictor")
st.caption(f"Trained on {len(df):,} apartment ads. All prices are asking prices, not final deal prices.")

tab1, tab2 = st.tabs(["Price estimate", "How accurate is it?"])

with tab1:
    c1, c2 = st.columns(2)
    location = c1.selectbox("Location", sorted(df["location"].unique()))
    rooms = c2.number_input("Rooms", 1, 10, 3)
    area = c1.number_input("Size (m²)", 12.0, 600.0, 90.0, step=1.0)
    floor = c2.number_input("Floor", 1, 34, 5)
    building_floors = c1.number_input("Building height (floors)", 1, 34, 9)
    new_building = c2.radio("New building?", ["Yes", "No"], horizontal=True) == "Yes"
    repair = c1.radio("Renovated?", ["Yes", "No"], horizontal=True) == "Yes"
    st.radio("Bill of sale?", ["Yes", "No"], horizontal=True)
    st.radio("Mortgage possible?", ["Yes", "No"], horizontal=True)
    st.caption("Bill of sale and mortgage are asked but ignored: they barely move the price.")

    if st.button("Estimate price", type="primary"):
        if floor > building_floors:
            st.error("The flat cannot be on a higher floor than the building has.")
        else:
            row = pd.DataFrame(
                [{
                    "rooms": rooms, "area": area, "floor": floor,
                    "building_floors": building_floors,
                    "new_building": int(new_building), "repair": int(repair),
                    "room_area": area / rooms, "location": location,
                }]
            )
            est = model.predict(make_features(row, columns))[0]
            st.success(f"Estimated asking price: **{est:,.0f} AZN**")
            st.write(f"That is about **{est / area:,.0f} AZN per m²**.")
            st.caption("Most reliable for normal 2–3 room flats; be careful with mansions or tiny studios.")

with tab2:
    st.write(
        "We hid 20% of the ads from the model while it learned, then asked it to guess their prices."
    )
    a, b, c = st.columns(3)
    a.metric("Typical miss", f"{stats['median_err']:.0%}")
    b.metric("Within 10% of the ad", f"{stats['within10']:.0%}")
    c.metric("Within 20% of the ad", f"{stats['within20']:.0%}")
    st.caption(f"Checked on {stats['n_test']:,} unseen ads. Share of price variation explained (R²): {stats['r2']:.2f}.")
