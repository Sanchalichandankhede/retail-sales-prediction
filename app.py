import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoder
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Retail Sales Prediction",
    page_icon="🛒",
    layout="centered"
)
st.title("🛒 Retail Sales Prediction")
st.write("Predict retail sales using the trained Random Forest model.")

st.divider()

store_id = st.selectbox(
    "Select Store",
    ["store_1", "store_2", "store_3", "store_4", "store_5"]
)

item_id = st.text_input(
    "Item ID",
    value="item_1"
)

price = st.number_input(
    "Price",
    min_value=0.0,
    value=20.0,
    step=0.1
)

promo = st.selectbox(
    "Promotion",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

weekday = st.selectbox(
    "Weekday",
    [0, 1, 2, 3, 4, 5, 6]
)

month = st.selectbox(
    "Month",
    list(range(1, 13))
)

# Prediction

if st.button("Predict Sales", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "store_id": [store_id],
        "item_id": [item_id],
        "price": [price],
        "promo": [promo],
        "weekday": [weekday],
        "month": [month]
    })

    # Encode categorical variables
    encoded_data = encoder.transform(
        input_data[["store_id", "item_id"]]
    )

    encoded_df = pd.DataFrame(
        encoded_data.toarray(),
        columns=encoder.get_feature_names_out(
            ["store_id", "item_id"]
        )
    )

    # Numerical features
    numerical_data = input_data[
        ["price", "promo", "weekday", "month"]
    ].reset_index(drop=True)

    # Combine encoded and numerical features
    final_input = pd.concat(
        [numerical_data, encoded_df],
        axis=1
    )

    # Make sure feature order matches training data
    if hasattr(model, "feature_names_in_"):
        final_input = final_input.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    # Prediction
    prediction = model.predict(final_input)[0]

    # Display result
    st.success("Prediction completed!")

    st.metric(
        label="Predicted Sales",
        value=f"{prediction:.2f} units"
    )

    st.info(
        f"The predicted sales for {item_id} at {store_id} "
        f"are approximately **{prediction:.2f} units**."
    )