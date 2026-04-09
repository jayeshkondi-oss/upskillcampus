import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.sidebar.title("🌱 About Project")
st.sidebar.write("This project predicts crop yield using machine learning based on cost and location factors.")

# -------------------------
# TITLE
# -------------------------
st.title("🌾 Agriculture Crop Yield Prediction")
st.markdown("### Predict crop yield using Machine Learning (Random Forest Model)")


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("datafile.csv")

df.columns = df.columns.str.strip()
st.write(df.columns) 
st.subheader("Dataset Preview")
st.write(df.head())

st.markdown("### 📊 Dataset Information")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -------------------------
# PREPROCESSING
# -------------------------
df = df.dropna()

le_crop = LabelEncoder()
le_state = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['State'] = le_state.fit_transform(df['State'])

# Target = Yield
X = df[['Crop', 'State', 
        'Cost of Cultivation (`/Hectare) A2+FL',
        'Cost of Cultivation (`/Hectare) C2',
        'Cost of Production (`/Quintal) C2']]

y = df['Yield (Quintal/ Hectare)']
    
# -------------------------
# MODEL
# -------------------------
if os.path.exists("model.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    pickle.dump(model, open("model.pkl", "wb"))

# -------------------------
# USER INPUT
# -------------------------
st.subheader("Enter Details")

crop = st.selectbox("Crop", le_crop.classes_)
state = st.selectbox("State", le_state.classes_)

cost_cultivation = st.number_input("Cost of Cultivation")
cost_c2 = st.number_input("Cost of Cultivation C2")
cost_production = st.number_input("Cost of Production")

crop_enc = le_crop.transform([crop])[0]
state_enc = le_state.transform([state])[0]

input_data = np.array([[crop_enc, state_enc, cost_cultivation, cost_c2, cost_production]])

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):
    result = model.predict(input_data)
    st.success(f"🌾 Estimated Yield: {result[0]:.2f} Quintal/Hectare")
st.markdown("---")
st.subheader("📈 Data Visualization")

# Crop-wise yield
if st.checkbox("Show Average Yield by Crop"):
    crop_data = df.groupby("Crop")["Yield (Quintal/ Hectare)"].mean()
    st.bar_chart(crop_data)

# State-wise yield
if st.checkbox("Show Average Yield by State"):
    state_data = df.groupby("State")["Yield (Quintal/ Hectare)"].mean()
    st.bar_chart(state_data)