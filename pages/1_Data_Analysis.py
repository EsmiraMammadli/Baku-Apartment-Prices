import numpy as np
import pandas as pd
import streamlit as st

from utils import load_clean, load_data

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")
raw = load_data()
df = load_clean()


def card(assumed, found, decided):
    a, b, c = st.columns(3)
    a.info(f"**Assumed**\n\n{assumed}")
    b.warning(f"**Found**\n\n{found}")
    c.success(f"**Decided**\n\n{decided}")


st.title("📊 Data Analysis")
st.caption(f"{len(raw):,} ads · {raw['location'].nunique()} locations · all numbers below are calculated live from the data")

# ---------------------------------------------------------------- 1
st.header("1 · The typical price")
typical, average = raw["price"].median(), raw["price"].mean()
c1, c2 = st.columns([2, 1])
with c1:
    capped = raw[raw["price"] <= 1_000_000]["price"]
    counts, edges = np.histogram(capped, bins=20, range=(0, 1_000_000))
    hist = pd.DataFrame(
        {"ads": counts}, index=[f"{int(e / 1000):04d}k" for e in edges[:-1]]
    )
    st.bar_chart(hist)
    st.caption("Asking prices up to 1 million AZN (bars show how many ads fall in each 50k range).")
with c2:
    st.metric("Typical flat (middle)", f"{typical:,.0f} AZN")
    st.metric("Average", f"{average:,.0f} AZN")
    st.metric("Most expensive ad", f"{raw['price'].max():,.0f} AZN")
card(
    "The average shows what a typical flat costs.",
    f"Average: {average:,.0f} AZN. Typical flat: {typical:,.0f}. A few luxury ads pull the average up.",
    "Use the typical price, not the average.",
)

# ---------------------------------------------------------------- 2
st.header("2 · Not every ad is a normal flat")
a, b, c, d = st.columns(4)
a.metric("Ads with 8+ rooms", int((raw["rooms"] >= 8).sum()))
b.metric("Flats larger than 400 m²", int((raw["area"] > 400).sum()))
c.metric("Ads above 1 million AZN", int((raw["price"] > 1_000_000).sum()))
d.metric("Flats smaller than 25 m²", int((raw["area"] < 25).sum()))
share = raw["rooms"].isin([2, 3]).mean()
card(
    "Every ad describes a typical flat.",
    "Some are mansions, tiny studios, or maybe typing mistakes (one ad even lists 345,000 m²).",
    f"Keep the real ones in view; remove impossible values. {share:.0%} of ads are 2 or 3 rooms.",
)

# ---------------------------------------------------------------- 3
st.header("3 · Bigger flats cost more, so compare per m²")
bins = [0, 50, 70, 90, 120, 160, 1600]
labels = ["1) up to 50 m²", "2) 50-70", "3) 70-90", "4) 90-120", "5) 120-160", "6) 160+"]
s = df.assign(size=pd.cut(df["area"], bins, labels=labels))
g = s.groupby("size", observed=True).agg(price=("price", "median"), price_m2=("price_m2", "median"))
c1, c2 = st.columns(2)
c1.subheader("Typical total price (AZN)")
c1.bar_chart(g["price"])
c2.subheader("Typical price per m² (AZN)")
c2.bar_chart(g["price_m2"])
card(
    "Size drives the price, so flats must be compared fairly.",
    f"Total price grows {g['price'].iloc[-1] / g['price'].iloc[0]:.1f}x from the smallest to the largest group, "
    f"but per m² it stays between {g['price_m2'].min():,.0f} and {g['price_m2'].max():,.0f} AZN.",
    "Compare flats by price per m². Size stays a key detail.",
)

# ---------------------------------------------------------------- 4
st.header("4 · The same m² costs very different amounts by place")
loc = df.groupby("location").agg(ads=("price_m2", "size"), price_m2=("price_m2", "median"))
big = loc[loc["ads"] >= 100].sort_values("price_m2", ascending=False)
top, bottom = big.head(6), big.tail(6).sort_values("price_m2", ascending=False)
show = pd.concat([top, bottom])
show.index = [f"{i:02d} {n}" for i, n in enumerate(show.index, 1)]
st.bar_chart(show["price_m2"], horizontal=True)
rare = loc[loc["ads"] < 50]
card(
    "Where a flat stands changes its price a lot.",
    f"Per m², the priciest area costs {big['price_m2'].max() / big['price_m2'].min():.1f}x the cheapest "
    "(only places with at least 100 ads are compared).",
    f"Group places by type. Put the {len(rare)} rarely listed places (under 50 ads each, "
    f"{rare['ads'].sum() / len(df):.1%} of the data) together as “Others”.",
)

# ---------------------------------------------------------------- 5
st.header("5 · Higher floors look pricier. Look closer")
fb = [0, 3, 6, 9, 12, 40]
fl = ["1) floors 1-3", "2) 4-6", "3) 7-9", "4) 10-12", "5) 13+"]
f = df.assign(fg=pd.cut(df["floor"], fb, labels=fl)).groupby("fg", observed=True).agg(
    price=("price", "median"), price_m2=("price_m2", "median"), new=("new_building", "mean")
)
c1, c2, c3 = st.columns(3)
c1.subheader("Typical price (AZN)")
c1.bar_chart(f["price"])
c2.subheader("Price per m² (AZN)")
c2.bar_chart(f["price_m2"])
c3.subheader("Share in new buildings")
c3.bar_chart(f["new"])
card(
    "Higher floors should cost more because of the view.",
    f"Total price climbs with the floor, but price per m² stays almost flat. "
    f"On floor 13+, {f['new'].iloc[-1]:.0%} of flats are in new buildings.",
    "Never judge by floor alone. Look at the building too.",
)

# ---------------------------------------------------------------- 6
st.header("6 · New buildings cost more. Or do they?")
n = df.groupby("new_building").agg(price=("price", "median"), area=("area", "median"), price_m2=("price_m2", "median"))
n.index = ["Old building", "New building"]
c1, c2, c3 = st.columns(3)
c1.subheader("Typical price (AZN)")
c1.bar_chart(n["price"])
c2.subheader("Typical size (m²)")
c2.bar_chart(n["area"])
c3.subheader("Price per m² (AZN)")
c3.bar_chart(n["price_m2"])
card(
    "New buildings are far more expensive.",
    f"Total price is {n['price'].iloc[1] / n['price'].iloc[0]:.1f}x higher, but new flats are bigger too "
    f"({n['area'].iloc[1]:.0f} vs {n['area'].iloc[0]:.0f} m²). Per m² the gap is "
    f"{n['price_m2'].iloc[1] / n['price_m2'].iloc[0] - 1:.0%}.",
    "Always read “new building” together with size.",
)

# ---------------------------------------------------------------- 7
st.header("7 · Every detail matters? Two say very little")
corr = (
    df[["price", "area", "rooms", "floor", "building_floors", "new_building", "repair", "bill_of_sale", "mortgage"]]
    .corr(method="spearman")["price"]
    .drop("price")
    .abs()
    .sort_values(ascending=False)
)
names = {
    "area": "Size", "rooms": "Rooms", "floor": "Floor", "building_floors": "Building height",
    "new_building": "New building", "repair": "Renovation", "bill_of_sale": "Bill of sale", "mortgage": "Mortgage",
}
shown = corr.rename(index=names)
shown.index = [f"{i} {n}" for i, n in enumerate(shown.index, 1)]
st.bar_chart(shown, horizontal=True)
st.caption("How closely each detail follows the price (0 = no link, 1 = perfect link). Location is a list of places, so it has its own section above.")
card(
    "Every detail in an ad matters for the price.",
    "A bill of sale or a mortgage option is almost unrelated to the price.",
    "Leave both out of the estimate. The app still asks, but ignores them.",
)
