import pandas as pd
import streamlit as st

from utils import load_clean, load_data

st.set_page_config(page_title="Baku Apartment Prices", page_icon="🏙️", layout="wide")

raw = load_data()
df = load_clean()

st.title("🏙️ What apartment ads reveal about Baku")
st.subheader("The seller names a price. Is it fair?")
st.write(
    "To answer this, we read thousands of apartment ads and tested every assumption we had. "
    "Some held, some broke. This app shows both: the analysis behind our decisions, and a price estimator built on them."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Apartment ads", f"{len(raw):,}")
c2.metric("Locations in Baku", f"{raw['location'].nunique()}")
c3.metric("Typical asking price", f"{raw['price'].median():,.0f} AZN")
c4.metric("Ads with 2 or 3 rooms", f"{raw['rooms'].isin([2, 3]).mean():.0%}")

st.info(
    "👈 Use the sidebar. **Data Analysis** walks through our seven assumptions. "
    "**Price Prediction** estimates a flat's price and shows how accurate it is."
)

st.markdown("### Seven decisions and the reasons behind them")
st.table(
    pd.DataFrame(
        {
            "Decision": [
                "Use the typical price, not the average",
                "Compare flats per m²",
                "Group rarely listed places as “Others”",
                "Group places by type: metro, district, outer",
                "Treat floor and building height separately",
                "Look at room size, not only room count",
                "Ignore bill of sale and mortgage",
            ],
            "Reason": [
                "A few luxury ads stretch the average",
                "Size is the biggest driver of price",
                "Too few ads to trust one by one",
                "Outer settlements cost less per m²",
                "Each says something different",
                "Separates spacious flats from cramped ones",
                "Almost no link to price",
            ],
        }
    ).set_index("Decision")
)

st.caption(
    "All prices are asking prices, not final deal prices: sellers may ask more than buyers finally pay. "
    "Three habits for reading any data: question the obvious, compare like with like, watch for hidden links."
)
