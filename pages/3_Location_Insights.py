import os

import pandas as pd
import streamlit as st

from utils import load_clean, location_type

st.set_page_config(page_title="Location Insights", page_icon="📍", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data
def read_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def money(x):
    return f"{x:,.0f} AZN"


df = load_clean()

# Average price and average price per m² for every place (computed live from the ads).
places = df.groupby("location").agg(
    ads=("price", "size"),
    avg_price=("price", "mean"),
    avg_m2=("price_m2", "mean"),
)
places["type"] = places.index.map(location_type)
type_names = {"metro": "Metro areas", "district": "Districts", "settlement": "Settlements"}

st.title("📍 Location Insights")
st.caption(
    "Averages by place, rooms, floors and building age. "
    "Place names end with m. (metro area), r. (district) or q. (settlement). All prices are asking prices in AZN."
)

# ---------------------------------------------------------------- 1
st.header("1 · Places at a glance")
counts = places["type"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Places in the dataset", len(places))
c2.metric("Metro areas", int(counts.get("metro", 0)))
c3.metric("Districts", int(counts.get("district", 0)))
c4.metric("Settlements", int(counts.get("settlement", 0)))

rows = []
for kind in ("district", "metro", "settlement"):
    sub = places[places["type"] == kind]
    rows.append(
        {
            "Type": type_names[kind],
            "Cheapest on average": sub["avg_price"].idxmin(),
            "Lowest average price (AZN)": sub["avg_price"].min(),
            "Priciest on average": sub["avg_price"].idxmax(),
            "Highest average price (AZN)": sub["avg_price"].max(),
        }
    )
st.dataframe(
    pd.DataFrame(rows).set_index("Type"),
    column_config={
        "Lowest average price (AZN)": st.column_config.NumberColumn(format="%.0f"),
        "Highest average price (AZN)": st.column_config.NumberColumn(format="%.0f"),
    },
    use_container_width=True,
)

# ---------------------------------------------------------------- 2
st.header("2 · How far is each place from the cheapest one?")
st.write(
    "Within each type of place, we divide the average price by the lowest average price of that type. "
    "A value of 3.0x means the average flat costs three times as much as in the cheapest place."
)

settle = read_csv("settlements_vs_minimum.csv")
dist = read_csv("districts_vs_minimum.csv")
mult = "Multiple of Minimum (x)"
price = "Average Price (AZN)"

tab_s, tab_d = st.tabs(["Settlements", "Districts"])
with tab_s:
    low, high = settle.loc[settle[price].idxmin()], settle.loc[settle[price].idxmax()]
    st.write(
        f"The cheapest settlement is **{low['Place']}** ({money(low[price])}). "
        f"The priciest is **{high['Place']}** at **{high[mult]:.1f}x** the minimum."
    )
    st.bar_chart(settle.set_index("Place")[mult], horizontal=True, height=900)
    st.dataframe(
        settle,
        column_config={price: st.column_config.NumberColumn(format="%.0f"), mult: st.column_config.NumberColumn(format="%.1fx")},
        hide_index=True,
        use_container_width=True,
    )
with tab_d:
    low, high = dist.loc[dist[price].idxmin()], dist.loc[dist[price].idxmax()]
    st.write(
        f"The cheapest district is **{low['Place']}** ({money(low[price])}). "
        f"The priciest is **{high['Place']}** at **{high[mult]:.1f}x** the minimum."
    )
    st.bar_chart(dist.set_index("Place")[mult], horizontal=True)
    st.dataframe(
        dist,
        column_config={price: st.column_config.NumberColumn(format="%.0f"), mult: st.column_config.NumberColumn(format="%.1fx")},
        hide_index=True,
        use_container_width=True,
    )

metro = places[places["type"] == "metro"]
metro_top = metro["avg_price"].idxmax()
st.caption(
    f"For metro areas the same comparison gives {metro_top} at "
    f"{metro['avg_price'].max() / metro['avg_price'].min():.1f}x the cheapest metro area ({metro['avg_price'].idxmin()})."
)

# ---------------------------------------------------------------- 3
st.header("3 · Rooms and price")
ads = read_csv("rooms_ad_counts.csv")
stats = read_csv("rooms_price_stats.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Most rooms in one ad", int(ads["Rooms"].max()))
c2.metric("Fewest rooms in one ad", int(ads["Rooms"].min()))
c3.metric("Most common", f"{int(ads.loc[ads['Ads'].idxmax(), 'Rooms'])} rooms")

left, right = st.columns(2)
left.subheader("Number of ads by rooms")
left.bar_chart(ads.set_index("Rooms")["Ads"])
right.subheader("Average price by rooms (AZN)")
right.bar_chart(stats.set_index("Rooms")[price])

rises = bool(stats[price].is_monotonic_increasing)
st.write(
    "As the room count grows, the average price grows too"
    + (" in every step." if rises else ", with a few exceptions.")
    + " Size, location and other factors explain most of the differences. "
    "Prices also vary a lot inside each room count: the coefficient of variation shows how widely they spread."
)
st.dataframe(
    stats,
    column_config={
        "Price Std Dev (AZN)": st.column_config.NumberColumn(format="%.0f"),
        price: st.column_config.NumberColumn(format="%.0f"),
        "Coefficient of Variation (%)": st.column_config.NumberColumn(format="%.1f%%"),
    },
    hide_index=True,
    use_container_width=True,
)
st.caption("The stats table covers room counts from 1 to 8. Larger flats have only a handful of ads each.")

# ---------------------------------------------------------------- 4
st.header("4 · Price per m²: the priciest and cheapest places")
top, bottom = places["avg_m2"].idxmax(), places["avg_m2"].idxmin()
c1, c2 = st.columns(2)
c1.metric(f"Priciest m²: {top}", f"{places['avg_m2'].max():,.2f} AZN")
c2.metric(f"Cheapest m²: {bottom}", f"{places['avg_m2'].min():,.2f} AZN")
st.write(
    f"One m² in **{top}** costs **{places['avg_m2'].max() / places['avg_m2'].min():.1f}x** as much as in **{bottom}**."
)

# ---------------------------------------------------------------- 5
st.header("5 · Floors and building age")
st.write(
    "Higher flats would normally be expected to cost less, yet average prices rise with the floor. "
    "High floors are mostly in new buildings, so floor and building age are tangled together. "
    "Switch the filter between all flats and new buildings to see the effect."
)
choice = st.radio("Buildings", ["All", "New buildings", "Old buildings"], horizontal=True)
f = df[df["floor"] <= 20]
if choice == "New buildings":
    f = f[f["new_building"] == 1]
elif choice == "Old buildings":
    f = f[f["new_building"] == 0]
by_floor = f.groupby("floor").agg(avg_price=("price", "mean"), ads=("price", "size"))
st.bar_chart(by_floor["avg_price"])
st.caption("Average asking price (AZN) by floor, floors 1 to 20.")
spread = by_floor["avg_price"].std() / by_floor["avg_price"].mean()
st.write(
    f"Spread between floors: **{spread:.0%}** of the average price. "
    "The curve is flatter for new buildings alone than for all flats together, "
    "so part of the climb with the floor comes from building age, not height."
)
if choice == "Old buildings":
    st.caption("Old buildings have very few ads above floor 10, so those bars are unreliable.")

mean_by_age = df.groupby("new_building")["price"].mean()
st.subheader("New buildings vs old buildings")
st.metric("New / old average price", f"{mean_by_age.loc[1] / mean_by_age.loc[0]:.2f}x")
st.write(
    "Several things push new buildings up: they are newer, taller and usually larger. "
    "Low floors do not show a sharp advantage, because low buildings include many old 4 to 5 storey blocks."
)

# ---------------------------------------------------------------- 6
st.header("6 · New vs old prices, place by place")
nv = read_csv("new_vs_old_price_by_place.csv")
ratio, diff = "New/Old Ratio (x)", "New minus Old (AZN)"
hi, lo = nv.loc[nv[ratio].idxmax()], nv.loc[nv[ratio].idxmin()]
older = nv[nv[diff] < 0]
c1, c2, c3 = st.columns(3)
c1.metric(f"Biggest gap: {hi['Place']}", f"{hi[ratio]:.2f}x")
c2.metric(f"Smallest gap: {lo['Place']}", f"{lo[ratio]:.2f}x")
c3.metric("Places where old costs more", len(older))
st.write(
    f"In **{len(older)}** places ({', '.join(older['Place'])}) the average old building costs more than the average new one. "
    "Rooms and size play a role here too, and some places have very few ads of one kind."
)
st.bar_chart(nv.set_index("Place")[ratio], horizontal=True, height=1200)
st.dataframe(
    nv,
    column_config={
        "Old Building Avg (AZN)": st.column_config.NumberColumn(format="%.0f"),
        "New Building Avg (AZN)": st.column_config.NumberColumn(format="%.0f"),
        ratio: st.column_config.NumberColumn(format="%.2fx"),
        diff: st.column_config.NumberColumn(format="%.0f"),
    },
    hide_index=True,
    use_container_width=True,
)
st.caption("Only places that have both new and old building ads are listed.")

# ---------------------------------------------------------------- 7
st.header("7 · How many ads are new or old?")
cnt = read_csv("new_vs_old_ad_counts.csv")
new_top = cnt.loc[cnt["New Building Ads"].idxmax()]
old_top = cnt.loc[cnt["Old Building Ads"].idxmax()]
c1, c2 = st.columns(2)
c1.metric(f"Most new building ads: {new_top['Place']}", int(new_top["New Building Ads"]))
c2.metric(f"Most old building ads: {old_top['Place']}", int(old_top["Old Building Ads"]))

new_pct, old_pct = "New Building Share (%)", "Old Building Share (%)"
all_new = cnt.loc[cnt[new_pct] == cnt[new_pct].max(), "Place"]
all_old = cnt.loc[cnt[old_pct] == cnt[old_pct].max(), "Place"]
st.write(
    f"**Highest share of new building ads ({cnt[new_pct].max()}%), {len(all_new)} places:** {', '.join(all_new)}."
)
st.write(
    f"**Highest share of old building ads ({cnt[old_pct].max()}%), {len(all_old)} places:** {', '.join(all_old)}."
)
st.caption("Many of these places have only a few ads, so a 100% share can rest on a single listing.")
st.dataframe(
    cnt,
    column_config={
        new_pct: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
        old_pct: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
    },
    hide_index=True,
    use_container_width=True,
)
