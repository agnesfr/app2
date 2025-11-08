# "Chat GPT and copilot was used in this code, chat was used with seesions state to make the predict button and "main page work" aswell as making the input data match training data
# aswell as debugging throughtout the processen and in the end rename variables and making the code look nicer and the text be the right size and color"

import streamlit as st
import pandas as pd
import numpy as np
import pickle

#All colums with dummies in trainings set as global variables

months_colums= [
    "month_April","month_August","month_December","month_February",
    "month_January","month_July","month_June","month_March","month_May",
    "month_November","month_October","month_September"
]

days_colums = [
    "dayofweek_Friday","dayofweek_Monday","dayofweek_Saturday","dayofweek_Sunday",
    "dayofweek_Thursday","dayofweek_Tuesday","dayofweek_Wednesday"
]

hours_colums = [
    "hour_0","hour_1","hour_2","hour_3","hour_4","hour_5","hour_6","hour_7",
    "hour_8","hour_9","hour_10","hour_11","hour_12","hour_13","hour_14","hour_15",
    "hour_16","hour_17","hour_18","hour_19","hour_20","hour_21","hour_22","hour_23"
]
weather_colums = [
    "weather_main_Clear", "weather_main_Clouds", "weather_main_Drizzle",
    "weather_main_Fog", "weather_main_Haze", "weather_main_Mist",
    "weather_main_Rain", "weather_main_Smoke", "weather_main_Snow",
    "weather_main_Squall", "weather_main_Thunderstorm",
]

holidays_colums = [
    "holiday_Christmas Day", "holiday_Columbus Day", "holiday_Independence Day",
    "holiday_Labor Day", "holiday_Martin Luther King Jr Day", "holiday_Memorial Day",
    "holiday_New Years Day", "holiday_State Fair", "holiday_Thanksgiving Day",
    "holiday_Veterans Day", "holiday_Washingtons Birthday",
]

#sorts features the same way as in traning csv
mapie_features = (
    ["temp", "rain_1h", "snow_1h", "clouds_all"]
    + months_colums
    + days_colums
    + hours_colums
    + weather_colums
    + holidays_colums
)

#Session state
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "X_locked" not in st.session_state:
    st.session_state.X_locked = None


#Cache functions
@st.cache_resource(show_spinner=False)
def load_mapie(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def fast_read_csv(file):
    return pd.read_csv(file, engine="pyarrow")


st.markdown("""
<h1 style="
    font-size: 64px;
    background: linear-gradient(90deg,orange, yellow, green);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    text-align: center;
">
Traffic Volume Predictor
</h1>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h4 style="text-align: center;">Utilize our advanced Machine Learning application to predict traffic volume</h4>
    """,
    unsafe_allow_html=True
)

st.image("traffic_image.gif", use_container_width=True)

st.sidebar.image("traffic_sidebar.jpg", use_container_width=True, caption="Traffic Volume Predictor")
st.sidebar.header("Input Features")
st.sidebar.subheader("You can either upload your data file or manually enter input features.")

# Option 1: Upload preview (informational)
sample_df = fast_read_csv("traffic_data_user.csv")
with st.sidebar.expander("Option 1: Upload CSV File"):
    st.write("Upload a CSV file containing traffic details.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    st.header("Sample Data format")
    st.dataframe(sample_df.head())
    st.warning("⚠️Ensure your uploaded file has the expected schema used by your model.")

# Option 2: Manual inputs
with st.sidebar.expander("Option 2: Fill Out Form"):
    st.write("Enter the traffic details manually using the form below.")
    holiday = st.selectbox( "Choose whether today is a designated holiday or not",options = [ "None","Christmas Day", "Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day", "Memorial Day", "New Years Day", "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"])
    avg_temp = st.number_input("Average Temperature in Kelvin", min_value=50.0, max_value=330.0, value=281.21, step=0.1)
    rain_1h = st.number_input("Amount in mm of rain that occurred in the hour", min_value=0.0, max_value=100.0, value=0.33, step=0.1)
    snow_1h = st.number_input("Amount in mm of snow that occurred in the hour", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    cloud_proc = st.number_input("Percentage of cloud cover", min_value=0, max_value=100, value=49, step=1)
    weather = st.selectbox("Choose current weather", options = [ "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"])
    month = st.selectbox("Choose month", options = ["January", "February", "March", "April", "May", "June", "July", "August","September", "October", "November", "December"])
    day = st.selectbox("Choose day of week", options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour = st.selectbox("Choose hour", options=list(range(1, 25)))  # 1..24 (we convert to 0..23 later)
    predict_pressed = st.button("Submit Form Data")

def df_input_to_format(df_raw: pd.DataFrame) -> pd.DataFrame:
    #makes csv file df with right colums
    df = df_raw.copy()

    for c in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors="coerce")
        else:
            ser = pd.Series(0, index=df.index)
        df[c] = ser.fillna(0)

    out = pd.DataFrame(0, index=df.index, columns=mapie_features)
    out["temp"] = df["temp"]
    out["rain_1h"] = df["rain_1h"]
    out["snow_1h"] = df["snow_1h"]
    out["clouds_all"] = df["clouds_all"]

    if "month" in df.columns:
        for idx, m in df["month"].astype(str).items():
            col = f"month_{m}"
            if col in months_colums:
                out.at[idx, col] = 1

    if "dayofweek" in df.columns:
        for idx, d in df["dayofweek"].astype(str).items():
            col = f"dayofweek_{d}"
            if col in days_colums:
                out.at[idx, col] = 1

    if "hour" in df.columns:
        h = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
        if h.between(1, 24).all():
            h = (h - 1).clip(0, 23)
        else:
            h = h.clip(0, 23)
        for idx, hv in h.items():
            col = f"hour_{hv}"
            if col in hours_colums:
                out.at[idx, col] = 1

    wsrc = "weather_main" if "weather_main" in df.columns else ("weather" if "weather" in df.columns else None)
    if wsrc:
        for idx, w in df[wsrc].fillna("").astype(str).items():
            col = f"weather_main_{w}"
            if col in weather_colums:
                out.at[idx, col] = 1

    if "holiday" in df.columns:
        for idx, hval in df["holiday"].fillna("None").astype(str).items():
            if hval != "None":
                col = f"holiday_{hval}"
                if col in holidays_colums:
                    out.at[idx, col] = 1

    out = out.reindex(columns=mapie_features, fill_value=0)
    return out

def predict_with_interval(mapie, X: pd.DataFrame, alpha: float):
    #Predicts point and interval
    output = mapie.predict(X, alpha=alpha)
    y_pred_mapie, y_pis = output

    point = np.asarray(y_pred_mapie, dtype=float).ravel()
    lower_bound   = np.asarray(y_pis[:, 0], dtype=float).ravel()
    higher_bound    = np.asarray(y_pis[:, 1], dtype=float).ravel()

    #Always bigger than zero
    point= np.maximum(point, 0)
    lower_bound    = np.maximum(lower_bound, 0)
    higher_bound    = np.maximum(higher_bound, 0)

    return point,lower_bound, higher_bound


if predict_pressed and uploaded is None:
    row = {
        "holiday": holiday,
        "temp": float(avg_temp),
        "rain_1h": float(rain_1h),
        "snow_1h": float(snow_1h),
        "clouds_all": int(cloud_proc),
        "weather_main": weather,
        "month": month,
        "weekday": day,
        "hour": hour
    }
    df_form = pd.DataFrame([row]) 
    st.session_state.X_locked = df_input_to_format(df_form)
    st.session_state.show_results = True
    st.success("✅ Form data submitted successfully")
elif uploaded is not None:
    st.success("✅ CSV File uploaded succesfully")
else:
    st.info("ℹ️ Please choose a data input method to proceed")

# Alpha slider
st.markdown("**Select alpha value for prediction intervals**")
alpha = st.slider("", min_value=0.01, max_value=0.50, value=0.26, step=0.01)
coverage = int(round((1 - alpha) * 100))

mapie = load_mapie("mapie_regressor.pkl")

if uploaded is not None:
    df_up = fast_read_csv(uploaded)
    X_up = df_input_to_format(df_up)
    y_point, y_lo, y_hi = predict_with_interval(mapie, X_up, alpha)

    out = df_up.copy()
    out["Predicted Volume"] = y_point.astype(int)
    out["Lower limit"] = y_lo.astype(int)
    out["Upper limit"] = y_hi.astype(int)

    st.subheader(f"Predictions Result with {coverage} % Prediction Interval")
    st.dataframe(out, use_container_width=True)

elif st.session_state.show_results and st.session_state.X_locked is not None:
    X_locked = st.session_state.X_locked
    y_point, y_lo, y_hi = predict_with_interval(mapie, X_locked, alpha)

    st.markdown(
        """
        <h2 style="margin:0 0 7px 0; font-weight:800;">
            Predicting Traffic Volume...
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style= font-size:14px; margin-bottom:4px;">
            Predicted Traffic Volume
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="font-size:45px; font-weight:700; line-height:1.1; margin-bottom:10px;">
            {int(y_point[0]):,}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="font-size:16px;">
            <b>Prediction Interval</b> ({coverage}%): [{int(y_lo[0]):,}, {int(y_hi[0]):,}]
        </div>
        """,
        unsafe_allow_html=True
    )


st.header("Model performance and inference")

tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predictive vs Actual", "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('Feature_Importance.png', caption="Relative importance of features in prediction.")

with tab2:
    st.write("### Histogram of Residuals")
    st.image('Residuals_Histogram.png', caption="Distribution of residuals to evaluate prediction quality.")

with tab3:
    st.write("### Predictive vs Actual")
    st.image('Predicted_vs_Actual.png', caption="Visual comparison of predicted and actual values.")

with tab4:
    st.write("### Coverage Plot")
    st.image('Prediction_Intervals.png', caption="Range of prediction with confidence intervals.")
