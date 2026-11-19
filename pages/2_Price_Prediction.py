import pandas as pd
import streamlit as st

from utils import load_clean, make_features, train_model

st.set_page_config(page_title="Price Prediction", page_icon="💰", layout="wide")
df = load_clean()
model, columns, stats = train_model()

st.title("💰 Price Prediction")
st.caption("All prices are asking prices, not final deal prices.")

tab1, tab2 = st.tabs(["Estimate a price", "How accurate is it?"])

with tab1:
    st.write("Write down your own guess first, then press the button.")
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
    st.caption("Bill of sale and mortgage are asked but ignored: they barely move the price. Try flipping them.")

    if st.button("Estimate price", type="primary"):
        if floor > building_floors:
            st.error("The flat cannot be on a higher floor than the building has.")
        else:
            row = pd.DataFrame(
                [{
                    "rooms": rooms, "area": area, "floor": floor, "building_floors": building_floors,
                    "new_building": int(new_building), "repair": int(repair),
                    "room_area": area / rooms, "location": location,
                }]
            )
            est = model.predict(make_features(row, columns))[0]
            st.success(f"Estimated asking price: **{est:,.0f} AZN**")
            st.write(f"That is about **{est / area:,.0f} AZN per m²**.")
            st.caption("Most reliable for normal 2-3 room flats; be careful with mansions or tiny studios.")

with tab2:
    st.write("We hid 20% of the ads while the model learned, then asked it to guess their prices.")
    a, b, c = st.columns(3)
    a.metric("Typical miss", f"{stats['median_err']:.0%}")
    b.metric("Within 10% of the ad", f"{stats['within10']:.0%}")
    c.metric("Within 20% of the ad", f"{stats['within20']:.0%}")
    st.caption(
        f"Checked on {stats['n_test']:,} unseen ads. Share of price variation explained (R²): {stats['r2']:.2f}. "
        "The model estimates what sellers ask, not what buyers finally pay."
    )
