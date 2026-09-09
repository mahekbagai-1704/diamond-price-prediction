import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
import pickle 
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score 

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide"
)


st.title('Diamond price prediction App💎')
st.markdown(
    "<h4 style='text-align:center; color:gray;'>Machine Learning based Diamond Price Prediction System</h4>",
    unsafe_allow_html=True
)
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to right,
        #dfe7fd,
        #edf2ff
    );
}

/* Main title */
h1 {
    color: #4F46E5;
    text-align: center;
    font-weight: bold;
}

/* Headers */
h2, h3 {
    color: #4338CA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #c7d2fe;
}

</style>
""", unsafe_allow_html=True)


df = pd.read_csv('diamonds.csv')

with open("diamond_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["Model"]
le_cut = data["le_cut"]
le_color = data["le_color"]
le_clarity = data["le_clarity"]
r2 = data["r2_score"]


# MODEL INFO FIRST
st.sidebar.subheader("🤖 Model Information")
st.sidebar.success(f"""
Model : Random Forest Regressor
R² Score : {r2:.2f}
""")
st.sidebar.divider()

# DATASET INFO
st.sidebar.subheader("📊 Dataset Information")
st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])
st.sidebar.metric(
    "Features Used",
    len(df.columns) - 1)

# EXPANDERS
with st.sidebar.expander("🔍 Preview Dataset"):
    st.dataframe(df.head())
with st.sidebar.expander("📈 Statistical Summary"):
    st.dataframe(df.describe())
with st.sidebar.expander("🧹 Null Values"):
    st.dataframe(df.isnull().sum())
with st.sidebar.expander("🔤 Label Encoding Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Cut")
        for i, label in enumerate(le_cut.classes_):
            st.write(f"{i} = {label}")
    with col2:
        st.write("Color")
        for i, label in enumerate(le_color.classes_):
            st.write(f"{i} = {label}")
    with col3:
        st.write("Clarity")
        for i, label in enumerate(le_clarity.classes_):
            st.write(f"{i} = {label}")

with st.expander("💎 Features Used for Prediction"):
    st.markdown("""

    - **Carat** → Weight of the diamond  
    - **Depth** → Overall depth percentage  
    - **Table** → Width of the diamond's top surface  
    - **Cut** → Quality of diamond cutting  
    - **Color** → Diamond color grading  
    - **Clarity** → Purity level of the diamond  
    - **X, Y, Z** → Diamond dimensions in mm  

    """)



st.subheader("💰 Predict Diamond Price")
st.info("Enter all diamond features carefully to get a more accurate price prediction.")
col1, col2 = st.columns(2)
with col1:
    carat = st.number_input(
        "Carat",
        value = 0.5
    )
    cut = st.selectbox(
        "Cut",
        le_cut.classes_
    )
    color = st.selectbox(
        "Color",
        le_color.classes_
    )
    clarity = st.selectbox(
        "Clarity",
        le_clarity.classes_
    )
    depth = st.number_input(
        "Depth",
        value = 61.5
    )

with col2:
    table = st.number_input(
        "Table",
        value = 55.0
    )
    x = st.number_input(
        "X Dimension",
        value = 5.0
    )
    y = st.number_input(
        "Y Dimension",
        value = 5.0
    )
    z = st.number_input(
        "Z Dimension",
        value = 3.0
    )

# ============================================
# PREDICTION
# ============================================

if st.button("✨ Predict Price", type="primary"):

    # Encode categorical inputs
    cut_encoded = le_cut.transform([cut])[0]
    color_encoded = le_color.transform([color])[0]
    clarity_encoded = le_clarity.transform([clarity])[0]

    # Create input array
    input_data = np.array([[
        carat,
        cut_encoded,
        color_encoded,
        clarity_encoded,
        depth,
        table,
        x,
        y,
        z
    ]])

    # Predict log(price)
    prediction_log = model.predict(input_data)

    # Convert back to original price
    prediction_price = np.exp(prediction_log)

    # Display result
    st.success(
        f"💎 Predicted Diamond Price : ${prediction_price[0]:,.2f}"
    )