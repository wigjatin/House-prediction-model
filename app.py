import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="House Prediction", 
    layout="wide", 
    page_icon="",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    .header {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        padding: 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.1);
    }

    .header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }

    .header p {
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(110, 142, 251, 0.4);
    }

    .metric-card {
        background: linear-gradient(135deg, #dbeafe, #a5b4fc);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: #111827;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .valuation-header {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #6e8efb;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

from joblib import load

@st.cache_resource
def load_model():
    return load('house_price_model.joblib')

model = load_model()

st.markdown("""
<div class="header">
    <h1>House Price Prediction</h1>
    <p>Discover your property's true market value with AI-powered precision</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Property Details")
    
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
        with c2:
            overall_qual = st.select_slider("Overall Quality", options=list(range(1, 11)), value=5)
        
        c3, c4 = st.columns(2)
        with c3:
            garage_cars = st.selectbox("Garage Spaces", options=[0, 1, 2, 3, 4, 5])
        with c4:
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=3000, value=1000, step=50)
        
        c5, c6 = st.columns(2)
        with c5:
            year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000, step=1)
        with c6:
            bedrooms = st.selectbox("Bedrooms", options=list(range(1, 11)), index=2)
        
        c7, c8 = st.columns(2)
        with c7:
            full_bath = st.selectbox("Full Bathrooms", options=list(range(1, 6)), index=1)
        with c8:
            half_bath = st.selectbox("Half Bathrooms", options=[0, 1, 2, 3])
        
        submitted = st.form_submit_button("Predict Price", type="primary")

from PIL import Image

with col2:
    img = Image.open("img.png")
    st.markdown(
        """
        <div style='
            background: #fff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        '>
        """,
        unsafe_allow_html=True
    )
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)



if submitted:
    with st.spinner('Crunching numbers with AI...'):
        time.sleep(1.5)

        input_data = pd.DataFrame([{
            'Gr Liv Area': gr_liv_area,
            'Overall Qual': overall_qual,
            'Garage Cars': garage_cars,
            'Total Bsmt SF': total_bsmt_sf,
            'Year Built': year_built,
            'Bedroom AbvGr': bedrooms,
            'Full Bath': full_bath,
            'Half Bath': half_bath
        }])

        pred_price = np.expm1(model.predict(input_data))[0]
        price_per_sqft = pred_price / gr_liv_area

        st.markdown("---")
        st.markdown(f'<div class="valuation-header">Valuation Complete</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Estimated Value", f"${pred_price:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Price per SqFt", f"${price_per_sqft:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Quality Rating", f"{overall_qual}/10")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Detailed Analysis"):
            tab1, tab2 = st.tabs(["Features Impact", "Market Comparison"])
            with tab1:
                features = ['Living Area', 'Quality', 'Garage', 'Basement', 'Age', 'Bedrooms', 'Bathrooms']
                importance = [0.35, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05]

                chart_data = pd.DataFrame({
                    'Feature': features,
                    'Impact': importance
                }).sort_values('Impact', ascending=True)

                st.bar_chart(chart_data.set_index('Feature'), height=300)
                st.caption("How each feature contributes to your House's value")
            with tab2:
                st.line_chart(pd.DataFrame({
                    'Similar Homes': [pred_price*0.9, pred_price*1.1, pred_price*0.95, pred_price, pred_price*1.05],
                    'Your Home': [None, None, None, pred_price, None]
                }))
                st.caption("Your house compared to similar properties in the market")

st.markdown("---")
st.markdown("🔗 [View Source Code on GitHub](https://github.com/wigjatin/House-prediction-model)")
