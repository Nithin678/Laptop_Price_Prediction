import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import base64

# Load the dataset
df = pd.read_csv("C:\\Users\\DELL\\Downloads\\laptop_price_prediction\\laptop_data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("Designer.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Convert RAM to integer
df["Ram"] = df["Ram"].astype(str).str.replace('GB', '').astype('int32')

# Function to convert memory amount to MB
def turn_memory_amount_into_MB(value):
    if isinstance(value, float):
        return value
    if "GB" in value:
        return float(value[:value.find("GB")]) * 1000
    if "TB" in value:
        return float(value[:value.find("TB")]) * 1000000
    return None

df['Memory'] = df['Memory'].apply(turn_memory_amount_into_MB)

# Adding Touchscreen column
df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)

# Encoding OpsSys column
df = df.join(pd.get_dummies(df.OpSys))
df.drop('OpSys', axis=1, inplace=True)

# Model training
X = df[['Company', 'Ram']]
y = df['Price']

# Preprocessing the 'Company' column
column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_encoded = column_transformer.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# Streamlit app
st.title('Laptop Price Prediction')

company = st.selectbox('Select Company', df['Company'].unique())
ram = st.selectbox('Select RAM (GB)', df['Ram'].unique())

input_data = [[company, ram]]
input_df = pd.DataFrame(input_data, columns=['Company', 'Ram'])
input_encoded = column_transformer.transform(input_df)

if st.button('Check Price'):
    price = model.predict(input_encoded)[0]
    st.write(f"The estimated price of the laptop is {price:.2f} Euros")
