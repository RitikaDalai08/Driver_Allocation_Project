import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic

st.set_page_config(page_title="Smart Driver Allocation", layout="wide")

# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page width */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1300px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Deploy button and footer for a cleaner customer view */
    [data-testid="stAppDeployButton"],
    footer {
        display: none;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0b1220 0%, #1f2937 60%, #3b1d28 100%);
        border-radius: 22px;
        padding: 2.6rem 2rem;
        text-align: center;
        margin-bottom: 1.6rem;
        border-bottom: 4px solid #ff4b4b;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.82);
        max-width: 760px;
        margin: 0.8rem auto 0 auto;
        font-size: 1.02rem;
    }

    /* Section headings */
    h2 {
        font-weight: 600;
        font-size: 1.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-left: 4px solid #ff4b4b;
        border-radius: 14px;
        padding: 14px 18px;
    }

    /* Driver card */
    .driver-card {
        display: flex;
        align-items: center;
        gap: 16px;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .avatar {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff4b4b, #ff8a4b);
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .driver-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.65;
    }
    .driver-name {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Form without its own border */
    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    /* Find Driver button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #ff4b4b, #ff7a45);
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }
    [data-testid="stFormSubmitButton"] button p {
        color: #ffffff;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        opacity: 0.9;
        border: none;
    }

    /* Rounded map */
    iframe {
        border-radius: 14px;
    }

    [data-testid="stCaptionContainer"] {
        text-align: center;
    }

    /* Light pink to grey page background, white sheet in the middle */
    .stApp {
        background: linear-gradient(135deg, #f7d9e0 0%, #efdbe5 30%, #e1dfe9 65%, #d4d7de 100%) !important;
        background-attachment: fixed !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container,
    [data-testid="stMainBlockContainer"] {
        background: #ffffff;
        width: calc(100% - 3rem);
        max-width: 1300px;
        margin: 3.5rem auto 2rem auto;
        padding: 2.2rem 2.4rem 2.6rem 2.4rem;
        border-radius: 28px;
        box-shadow: 0 10px 40px rgba(60, 40, 60, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("ride_demand_model.pkl")

model = load_model()

@st.cache_data
def load_drivers():
    return pd.read_csv("drivers.csv").to_dict("records")

# =========================
# PAGE TITLE
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>Smart Driver Allocation & Ride Demand Prediction System</h1>
        <p>AI-powered vehicle hiring system using Machine Learning
        for ride demand prediction and intelligent driver allocation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# RIDE DEMAND PREDICTION
# =========================
with st.container(border=True):

    st.header("Ride Demand Prediction")

    slider_col1, slider_col2, slider_col3 = st.columns(3, gap="large")

    with slider_col1:
        hour = st.slider("Select Hour", 0, 23, 12)

    with slider_col2:
        day = st.slider("Select Day", 1, 31, 1)

    with slider_col3:
        weekday = st.slider("Select Weekday", 0, 6, 0)

    input_data = np.array([[hour, day, weekday]])
    prediction = model.predict(input_data)

    st.metric(
        "Predicted Ride Demand",
        f"{prediction[0]:.2f}"
    )

# =========================
# SESSION STATE
# =========================
if "nearest_driver" not in st.session_state:
    st.session_state.nearest_driver = None

if "distance" not in st.session_state:
    st.session_state.distance = None

if "customer_lat" not in st.session_state:
    st.session_state.customer_lat = 40.7580

if "customer_lon" not in st.session_state:
    st.session_state.customer_lon = -73.9855

# =========================
# DRIVER ALLOCATION
# =========================
with st.container(border=True):

    st.header("Nearest Driver Allocation")

    with st.form("driver_form"):

        input_col1, input_col2 = st.columns(2, gap="large")

        with input_col1:
            customer_lat = st.number_input(
                "Customer Latitude",
                value=40.7580
            )

        with input_col2:
            customer_lon = st.number_input(
                "Customer Longitude",
                value=-73.9855
            )

        submit = st.form_submit_button(
            "Find Driver"
        )

    # =========================
    # FIND DRIVER
    # =========================
    if submit:

        drivers = load_drivers()

        customer_location = (
            customer_lat,
            customer_lon
        )

        nearest_driver = min(
            drivers,
            key=lambda driver: geodesic(
                customer_location,
                (
                    driver["lat"],
                    driver["lon"]
                )
            ).km
        )

        distance = geodesic(
            customer_location,
            (
                nearest_driver["lat"],
                nearest_driver["lon"]
            )
        ).km

        st.session_state.nearest_driver = nearest_driver
        st.session_state.distance = distance
        st.session_state.customer_lat = customer_lat
        st.session_state.customer_lon = customer_lon

    # =========================
    # DISPLAY RESULT
    # =========================
    if st.session_state.nearest_driver is not None:

        nearest_driver = st.session_state.nearest_driver
        distance = st.session_state.distance

        customer_lat = st.session_state.customer_lat
        customer_lon = st.session_state.customer_lon

        st.markdown("---")

        info_col, map_col = st.columns([1, 2], gap="large")

        # Driver Details
        with info_col:

            st.subheader("Your Driver")

            st.markdown(
                f"""
                <div class="driver-card">
                    <div class="avatar">{nearest_driver['name'].split()[-1]}</div>
                    <div>
                        <div class="driver-label">Allocated Driver</div>
                        <div class="driver-name">{nearest_driver['name']}</div>
                        <span class="badge">&#9679; Available</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Distance to Customer",
                f"{distance:.2f} km"
            )

            st.metric(
                "Estimated Arrival Time",
                "5 Minutes"
            )

        # Map
        with map_col:

            st.subheader("Live Location")

            m = folium.Map(
                location=[
                    customer_lat,
                    customer_lon
                ],
                zoom_start=14
            )

            # Customer Marker
            folium.Marker(
                [
                    customer_lat,
                    customer_lon
                ],
                tooltip="Customer Location",
                icon=folium.Icon(
                    color="blue"
                )
            ).add_to(m)

            # Driver Marker
            folium.Marker(
                [
                    nearest_driver["lat"],
                    nearest_driver["lon"]
                ],
                tooltip=nearest_driver["name"],
                icon=folium.Icon(
                    color="red"
                )
            ).add_to(m)

            # Route Line
            folium.PolyLine(
                [
                    [
                        customer_lat,
                        customer_lon
                    ],
                    [
                        nearest_driver["lat"],
                        nearest_driver["lon"]
                    ]
                ],
                color="green",
                weight=4
            ).add_to(m)

            st_folium(
                m,
                use_container_width=True,
                height=520
            )

# =========================
# CONCLUSION
# =========================
with st.container(border=True):

    st.subheader("Conclusion")

    st.write(
        "This system uses Machine Learning to predict ride demand "
        "and Geospatial Analysis to allocate the nearest available driver. "
        "The solution helps reduce passenger waiting time and improves "
        "overall ride allocation efficiency."
    )

# =========================
# FOOTER
# =========================
st.caption(
    "Developed using Machine Learning, Streamlit, Folium, and Geospatial Driver Allocation."
)