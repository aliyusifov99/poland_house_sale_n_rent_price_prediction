import streamlit as st
import requests
import datetime

import os

# Get API URL from environment variable, default to localhost for local testing
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="Polish Housing Predictor", 
    page_icon="🏠",
    layout="wide"
)

# Custom CSS to make the UI look a bit cleaner
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    div.stButton > button:first-child {background-color: #0099ff; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🇵🇱 Polish Housing Price Predictor")
st.markdown("---")

# Sidebar for Mode Selection
st.sidebar.header("Configuration")
mode = st.sidebar.radio("I want to predict:", ["sale", "rent"], 
                        help="Choose 'sale' for purchase price or 'rent' for monthly rental cost.")

st.sidebar.info(
    f"""
    **Current Mode: {mode.upper()}**
    
    Fill in the details on the right to get an estimated market price based on historical data.
    """
)

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("1. Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox(
            "City", 
            ["szczecin", "warszawa", "krakow", "poznan", "gdansk", "wroclaw", "lodz", "gdynia", "bialystok", "bydgoszcz", "czestochowa", "katowice", "lublin", "radom", "rzeszow", "sosnowiec"],
            help="The city where the property is located."
        )
        type_ = st.selectbox(
            "Building Type", 
            ["blockOfFlats", "tenement", "apartmentBuilding"],
            help="Tenement: Old pre-war buildings. Block: Communist-era concrete. Apartment: Modern developments."
        )
        ownership = st.selectbox(
            "Ownership", 
            ["condominium", "cooperative", "municipal"],
            help="Legal ownership status of the property."
        )
        
    with col2:
        # VALIDATION: step=0.1 allows decimals, min_value prevents negative numbers
        square_meters = st.number_input(
            "Square Meters", 
            min_value=10.0, max_value=500.0, value=50.0, step=0.5,
            help="Total floor area of the apartment."
        )
        
        # VALIDATION: step=1 and format="%d" enforces INTEGERS (no 2.01 rooms)
        rooms = st.number_input(
            "Rooms", 
            min_value=1, max_value=10, value=2, step=1, format="%d",
            help="Number of rooms (excluding kitchen/bathroom)."
        )
        
        # VALIDATION: Year cannot be in the distant future
        current_year = datetime.datetime.now().year
        build_year = st.number_input(
            "Build Year", 
            min_value=1800, max_value=current_year+5, value=2000, step=1, format="%d",
            help="Year the building was constructed."
        )
    
    with col3:
        centre_dist = st.number_input(
            "Distance to Centre (km)", 
            min_value=0.0, value=2.0, step=0.1,
            help="Straight-line distance to the city center."
        )
        
        # VALIDATION: Floor and Floor Count integers
        floor = st.number_input(
            "Floor", 
            min_value=0, max_value=50, value=1, step=1, format="%d",
            help="The floor the apartment is on (0 = Ground Floor)."
        )
        floor_count = st.number_input(
            "Total Floors in Building", 
            min_value=1, max_value=50, value=5, step=1, format="%d",
            help="Total height of the building in floors."
        )

    st.subheader("2. Neighborhood Amenities (Distances in km)")
    st.caption("Distance to the nearest point of interest.")
    
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        school_dist = st.number_input("School", value=0.5, min_value=0.0, step=0.1)
        clinic_dist = st.number_input("Clinic", value=1.0, min_value=0.0, step=0.1)
        poi_count = st.number_input("POI Count (500m)", value=10, min_value=0, step=1, help="Number of amenities within 500m range.")
    with acol2:
        post_dist = st.number_input("Post Office", value=0.5, min_value=0.0, step=0.1)
        kindergarten_dist = st.number_input("Kindergarten", value=0.5, min_value=0.0, step=0.1)
        restaurant_dist = st.number_input("Restaurant", value=0.5, min_value=0.0, step=0.1)
    with acol3:
        college_dist = st.number_input("College", value=2.0, min_value=0.0, step=0.1)
        pharmacy_dist = st.number_input("Pharmacy", value=0.5, min_value=0.0, step=0.1)

    st.subheader("3. Key Features")
    feat_col1, feat_col2, feat_col3, feat_col4, feat_col5 = st.columns(5)
    with feat_col1:
        has_parking = st.checkbox("Parking Space", help="Assigned parking spot (underground or outside).")
    with feat_col2:
        has_balcony = st.checkbox("Balcony", help="Has a balcony or terrace.")
    with feat_col3:
        has_elevator = st.checkbox("Elevator", help="Building has a working elevator.")
    with feat_col4:
        has_security = st.checkbox("Security", help="Monitoring, closed area, or security guard.")
    with feat_col5:
        has_storage = st.checkbox("Storage Room", help="Basement or storage unit included.")

    # Submit Button
    submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)

if submitted:
    # --- LOGICAL SANITIZATION ---
    # User cannot be on floor 10 of a 4-story building
    if floor > floor_count:
        st.error(f"❌ Logical Error: You selected Floor {floor}, but the building only has {floor_count} floors.")
    else:
        # Construct the JSON payload
        payload = {
            "city": city,
            "type": type_,
            "squareMeters": square_meters,
            "rooms": rooms,
            "floor": floor,
            "floorCount": floor_count,
            "buildYear": build_year,
            "centreDistance": centre_dist,
            "poiCount": poi_count,
            "schoolDistance": school_dist,
            "clinicDistance": clinic_dist,
            "postOfficeDistance": post_dist,
            "kindergartenDistance": kindergarten_dist,
            "restaurantDistance": restaurant_dist,
            "collegeDistance": college_dist,
            "pharmacyDistance": pharmacy_dist,
            "ownership": ownership,
            "buildingMaterial": "brick",
            "condition": "unknown",
            "hasParkingSpace": int(has_parking),
            "hasBalcony": int(has_balcony),
            "hasElevator": int(has_elevator),
            "hasSecurity": int(has_security),
            "hasStorageRoom": int(has_storage)
        }
        
        try:
            with st.spinner(f"Calculating {mode} price..."):
                response = requests.post(f"{API_URL}/{mode}", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                price = result['predicted_price']
                
                st.success("Prediction Complete!")
                
                # Create a nice layout for the result
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(
                        label=f"Estimated {mode.title()} Price", 
                        value=f"{price:,.0f} PLN",
                        delta_color="normal"
                    )
                
                with res_col2:
                    price_per_m2 = price / square_meters
                    st.metric(
                        label="Price per m²", 
                        value=f"{price_per_m2:,.0f} PLN"
                    )
                
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not reach the backend. Is the API running?")