import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import requests
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="UPlanning - Smart Urban Development",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Beautiful CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global Styles */
    .main .block-container {
        font-family: 'Poppins', sans-serif;
        max-width: 1400px;
        padding-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .brand-tagline {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Modern color palette */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --light-gradient: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4facfe;
        --warning-color: #43e97b;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --text-light: #bdc3c7;
        --background-main: #f8f9fa;
        --background-white: #ffffff;
        --background-dark: #2c3e50;
        --border-color: #e9ecef;
        --shadow-soft: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-hover: 0 15px 35px rgba(31, 38, 135, 0.2);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        animation: fadeInUp 1s ease-out;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page selection
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = "Interactive Map"

# Navigation and Title
page_selection = st.sidebar.selectbox(
    "Navigate",
    ["Interactive Map", "ML Insights", "AI Recommendations", "City Layouts & Infrastructure", "Analytics"],
    index=["Interactive Map", "ML Insights", "AI Recommendations", "City Layouts & Infrastructure", "Analytics"].index(st.session_state.page_selection) if st.session_state.page_selection in ["Interactive Map", "ML Insights", "AI Recommendations", "City Layouts & Infrastructure", "Analytics"] else 0,
    key="sidebar_nav"
)

# Update session state when sidebar selection changes
if page_selection != st.session_state.page_selection:
    st.session_state.page_selection = page_selection

# Always show title
st.markdown('<h1 class="main-header">🏙️ UPlanning</h1>', unsafe_allow_html=True)
st.markdown('<p class="brand-tagline">Smart Urban Development Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# Modern Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# ML Model Selection
st.sidebar.subheader("🤖 Machine Learning Models")
use_clustering = st.sidebar.checkbox("✨ K-Means Clustering", value=True, 
                                    help="Group neighborhoods by sustainability characteristics")
use_prediction = st.sidebar.checkbox("🔮 Future Predictions", value=True,
                                    help="Predict greenery coverage 5 years into the future")
use_satellite = st.sidebar.checkbox("🛰️ Satellite Analysis", value=False,
                                   help="Advanced satellite imagery analysis (demo mode)")

# Weight sliders
st.sidebar.subheader("📊 Sustainability Weights")
st.sidebar.markdown("*Adjust factor importance in sustainability scoring:*")
greenery_weight = st.sidebar.slider("🌳 Greenery Coverage", 0.0, 1.0, 0.4, 0.05,
                                   help="Weight for green space and vegetation coverage")
density_weight = st.sidebar.slider("👥 Population Density (inverse)", 0.0, 1.0, 0.3, 0.05,
                                  help="Weight for population density (lower density = higher score)")
infra_weight = st.sidebar.slider("🏗️ Infrastructure Quality", 0.0, 1.0, 0.3, 0.05,
                                help="Weight for infrastructure development and quality")

# Normalize weights
total_weight = greenery_weight + density_weight + infra_weight
if total_weight > 0:
    greenery_weight /= total_weight
    density_weight /= total_weight
    infra_weight /= total_weight

# Data source selection
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Data Source")
data_source = st.sidebar.selectbox(
    "Select Data Source:",
    ["Sample Urban Data", "Upload Custom CSV", "Fetch OpenStreetMap Data", "Extended Urban Data"],
    help="Choose your data source for urban analysis"
)

# City selection for OSM
city_name = None
if data_source == "Fetch OpenStreetMap Data":
    city_name = st.sidebar.text_input("🏙️ Enter city name:", "Austin, Texas")
    if st.sidebar.button("🔄 Fetch Real Data"):
        with st.spinner("Fetching OpenStreetMap data..."):
            st.info("Note: Real OSM integration requires API setup. Using enhanced sample data for demo.")

# Analysis Focus
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Analysis Focus")
planning_mode = st.sidebar.radio(
    "Choose Analysis Type:",
    ["Overview Analysis", "City-Specific Planning", "Comparative Analysis"],
    help="Select the type of analysis you want to perform"
)

if planning_mode == "City-Specific Planning":
    st.sidebar.success("💡 Navigate to 'City Layouts & Infrastructure' for detailed recommendations")
elif planning_mode == "Comparative Analysis":
    st.sidebar.success("📊 Use the 'Analytics' tab to compare cities and neighborhoods")
else:
    st.sidebar.info("🔍 General overview mode - explore all features")

# Load sample urban data
@st.cache_data
def load_sample_urban_data():
    """Load sample urban data from CSV file"""
    try:
        df = pd.read_csv('sample_urban_data.csv')
        
        # Ensure required columns exist
        required_cols = ['neighborhood', 'latitude', 'longitude', 'greenery_percent', 
                        'population_density', 'infrastructure_score', 'air_quality_index', 
                        'traffic_congestion', 'public_transport_access']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in sample_urban_data.csv: {missing_cols}. Using fallback data.")
            return generate_fallback_data()
        
        # Validate data ranges
        if df.empty:
            st.warning("Sample urban data file is empty. Using fallback data.")
            return generate_fallback_data()
        
        # Clean and validate numeric columns
        numeric_cols = ['latitude', 'longitude', 'greenery_percent', 'population_density', 
                       'infrastructure_score', 'air_quality_index', 'traffic_congestion', 
                       'public_transport_access']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=numeric_cols)
        
        if df.empty:
            st.warning("No valid data found after cleaning. Using fallback data.")
            return generate_fallback_data()
        
        st.success(f"✅ Successfully loaded {len(df)} neighborhoods from sample urban data")
        return df
        
    except FileNotFoundError:
        st.info("📁 Sample urban data file not found. Using demonstration data.")
        return generate_fallback_data()
    except Exception as e:
        st.error(f"❌ Error loading sample urban data: {str(e)}. Using fallback data.")
        return generate_fallback_data()

# Load extended urban data
@st.cache_data
def load_extended_urban_data():
    """Load extended urban data from CSV file"""
    try:
        df = pd.read_csv('extended_urban_data.csv')
        
        # Ensure required columns exist
        required_cols = ['neighborhood', 'latitude', 'longitude', 'greenery_percent', 
                        'population_density', 'infrastructure_score', 'air_quality_index', 
                        'traffic_congestion', 'public_transport_access']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in extended_urban_data.csv: {missing_cols}. Using fallback data.")
            return generate_fallback_data()
        
        # Validate data ranges
        if df.empty:
            st.warning("Extended urban data file is empty. Using fallback data.")
            return generate_fallback_data()
        
        # Clean and validate numeric columns
        numeric_cols = ['latitude', 'longitude', 'greenery_percent', 'population_density', 
                       'infrastructure_score', 'air_quality_index', 'traffic_congestion', 
                       'public_transport_access']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=numeric_cols)
        
        if df.empty:
            st.warning("No valid data found after cleaning. Using fallback data.")
            return generate_fallback_data()
        
        st.success(f"✅ Successfully loaded {len(df)} neighborhoods from extended urban data")
        return df
        
    except FileNotFoundError:
        st.info("📁 Extended urban data file not found. Using demonstration data.")
        return generate_fallback_data()
    except Exception as e:
        st.error(f"❌ Error loading extended urban data: {str(e)}. Using fallback data.")
        return generate_fallback_data()

# Fallback data generation (in case extended data fails to load)
@st.cache_data
def generate_fallback_data():
    """Generate fallback sample data if extended data is unavailable"""
    np.random.seed(42)
    neighborhoods = [
        "Downtown Core", "Riverside District", "Green Valley", "Industrial Park",
        "Suburban East", "Tech Campus", "Historic Old Town", "Lakeside",
        "University Quarter", "Harbor View", "West End", "North Hills"
    ]
    
    n = len(neighborhoods)
    
    # Base coordinates (simulating a city)
    base_lat, base_lon = 40.7128, -74.0060
    
    data = {
        'neighborhood': neighborhoods,
        'latitude': [base_lat + np.random.uniform(-0.08, 0.08) for _ in range(n)],
        'longitude': [base_lon + np.random.uniform(-0.08, 0.08) for _ in range(n)],
        'greenery_percent': np.random.uniform(10, 80, n),
        'population_density': np.random.uniform(1000, 15000, n),
        'infrastructure_score': np.random.uniform(30, 95, n),
        'air_quality_index': np.random.uniform(20, 150, n),
        'traffic_congestion': np.random.uniform(10, 90, n),
        'public_transport_access': np.random.uniform(20, 95, n),
    }
    
    df = pd.DataFrame(data)
    
    # Add historical data for predictions (last 5 years)
    for year in range(5, 0, -1):
        col_name = f'greenery_{year}y_ago'
        # Simulate declining greenery over time
        df[col_name] = df['greenery_percent'] + np.random.uniform(5, 15, n)
    
    return df

# Calculate sustainability score
def calculate_sustainability_score(df, g_weight, d_weight, i_weight):
    """Calculate normalized sustainability score"""
    green_norm = df['greenery_percent'] / 100
    density_norm = 1 - (df['population_density'] / df['population_density'].max())
    infra_norm = df['infrastructure_score'] / 100
    
    score = (g_weight * green_norm + d_weight * density_norm + i_weight * infra_norm) * 100
    return score

# ML: K-Means Clustering
@st.cache_data
def perform_clustering(df, n_clusters=3):
    """Cluster neighborhoods by sustainability characteristics"""
    features = ['greenery_percent', 'population_density', 'infrastructure_score', 
                'air_quality_index', 'traffic_congestion']
    
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Assign meaningful labels
    cluster_labels = []
    for i in range(n_clusters):
        mask = clusters == i
        avg_green = df[mask]['greenery_percent'].mean()
        avg_density = df[mask]['population_density'].mean()
        
        if avg_green > 50:
            label = "🌳 Green & Sustainable"
        elif avg_density > 10000:
            label = "🏙️ High Density Urban"
        else:
            label = "⚠️ Needs Improvement"
        
        cluster_labels.append(label)
    
    df['cluster'] = clusters
    df['cluster_label'] = [cluster_labels[c] for c in clusters]
    
    return df, cluster_labels

# ML: Future Prediction
@st.cache_data
def predict_future_greenery(df):
    """Predict greenery coverage 5 years in the future"""
    # Look for historical columns in extended data format
    historical_cols = [col for col in df.columns if 'greenery_' in col and 'y_ago' in col]
    
    if not historical_cols:
        # If no historical data, return original df
        df['predicted_greenery_5y'] = df['greenery_percent']
        df['greenery_change'] = 0
        return df
    
    # Prepare training data
    predictions = []
    
    for idx, row in df.iterrows():
        try:
            # Get historical values in chronological order (5y ago to current)
            historical_values = []
            for year in [5, 4, 3, 2, 1]:
                col_name = f'greenery_{year}y_ago'
                if col_name in df.columns:
                    historical_values.append(row[col_name])
            
            # Add current value
            historical_values.append(row['greenery_percent'])
            
            # Simple linear regression for trend
            if len(historical_values) > 1:
                x_vals = list(range(len(historical_values)))
                coef = np.polyfit(x_vals, historical_values, 1)
                # Predict 5 years into the future
                future_pred = np.polyval(coef, len(historical_values) + 5)
                predictions.append(max(0, min(100, future_pred)))  # Clamp between 0-100
            else:
                predictions.append(row['greenery_percent'])
        except Exception as e:
            # Fallback to current value if prediction fails
            predictions.append(row['greenery_percent'])
    
    df['predicted_greenery_5y'] = predictions
    df['greenery_change'] = df['predicted_greenery_5y'] - df['greenery_percent']
    
    return df

# City-specific layout templates and infrastructure knowledge base
CITY_TEMPLATES = {
    "New York": {
        "climate": "humid_continental",
        "challenges": ["high_density", "air_pollution", "traffic", "limited_green_space"],
        "strengths": ["public_transport", "walkability", "mixed_use"],
        "layout_recommendations": {
            "green_corridors": "Implement linear parks along avenues to connect existing green spaces",
            "vertical_gardens": "Mandate green walls on buildings over 10 stories",
            "pocket_parks": "Convert underutilized lots into micro-parks (0.1-0.5 acres)",
            "rooftop_gardens": "Incentivize rooftop farming and recreation spaces"
        },
        "infrastructure_priorities": {
            "transit": "Expand subway accessibility and bike-share integration",
            "energy": "District cooling systems and solar panel mandates",
            "water": "Green infrastructure for stormwater management",
            "waste": "Pneumatic waste collection in dense areas"
        }
    },
    "Los Angeles": {
        "climate": "mediterranean",
        "challenges": ["sprawl", "car_dependency", "water_scarcity", "air_quality"],
        "strengths": ["year_round_growing", "solar_potential", "innovation_hubs"],
        "layout_recommendations": {
            "transit_oriented": "Develop high-density mixed-use around metro stations",
            "green_streets": "Convert wide boulevards to include median parks",
            "water_wise_landscaping": "Native plant corridors and xeriscaping",
            "solar_districts": "Concentrated solar installations with battery storage"
        },
        "infrastructure_priorities": {
            "transit": "Light rail expansion and bus rapid transit",
            "energy": "Distributed solar microgrids",
            "water": "Greywater recycling and rainwater harvesting",
            "waste": "Organic waste to energy facilities"
        }
    },
    "Chicago": {
        "climate": "continental",
        "challenges": ["harsh_winters", "industrial_legacy", "segregation"],
        "strengths": ["lakefront", "grid_system", "public_spaces"],
        "layout_recommendations": {
            "lakefront_extension": "Extend lakefront park system inland",
            "industrial_conversion": "Transform brownfields into eco-districts",
            "winter_resilience": "Enclosed walkways and underground connections",
            "equitable_access": "Ensure green space within 10-minute walk citywide"
        },
        "infrastructure_priorities": {
            "transit": "Modernize L system with climate-controlled stations",
            "energy": "District heating from renewable sources",
            "water": "Great Lakes water conservation and treatment",
            "waste": "Industrial symbiosis networks"
        }
    }
}

def get_city_specific_recommendations(city_name, neighborhood_data):
    """Generate city-specific sustainable layout and infrastructure recommendations"""
    
    # Default template for cities not in database
    default_template = {
        "climate": "temperate",
        "challenges": ["generic_urban_issues"],
        "strengths": ["existing_infrastructure"],
        "layout_recommendations": {
            "green_network": "Create interconnected green corridors",
            "mixed_use": "Promote 15-minute neighborhood concept",
            "density_balance": "Optimize density with quality of life",
            "climate_adaptation": "Design for local climate resilience"
        },
        "infrastructure_priorities": {
            "transit": "Sustainable mobility solutions",
            "energy": "Renewable energy integration",
            "water": "Water-sensitive urban design",
            "waste": "Circular economy principles"
        }
    }
    
    # Get city template or use default
    city_template = CITY_TEMPLATES.get(city_name, default_template)
    
    # Analyze neighborhood data to customize recommendations
    avg_density = neighborhood_data['population_density'].mean()
    avg_greenery = neighborhood_data['greenery_percent'].mean()
    avg_air_quality = neighborhood_data['air_quality_index'].mean()
    avg_traffic = neighborhood_data['traffic_congestion'].mean()
    
    # Generate customized recommendations
    recommendations = {
        "layout_strategy": generate_layout_strategy(city_template, avg_density, avg_greenery),
        "infrastructure_plan": generate_infrastructure_plan(city_template, neighborhood_data),
        "implementation_phases": generate_implementation_phases(city_template, neighborhood_data),
        "sustainability_metrics": generate_sustainability_targets(neighborhood_data)
    }
    
    return recommendations, city_template

def generate_layout_strategy(city_template, avg_density, avg_greenery):
    """Generate specific layout strategy based on city characteristics"""
    strategies = []
    
    # Density-based strategies
    if avg_density > 10000:
        strategies.append({
            "type": "High-Density Optimization",
            "description": "Vertical development with mandatory green space ratios",
            "specifics": [
                "15% ground-level green space minimum",
                "Rooftop gardens on 50% of buildings",
                "Vertical transportation hubs",
                "Mixed-use towers with community spaces"
            ]
        })
    elif avg_density < 3000:
        strategies.append({
            "type": "Smart Densification",
            "description": "Gentle density increase with green infrastructure",
            "specifics": [
                "Missing middle housing types",
                "Green corridors connecting neighborhoods",
                "Transit-oriented development nodes",
                "Preserve existing tree canopy"
            ]
        })
    
    # Greenery-based strategies
    if avg_greenery < 30:
        strategies.append({
            "type": "Green Infrastructure Expansion",
            "description": "Aggressive greening program with multiple typologies",
            "specifics": [
                "Street tree planting program (1000 trees/year)",
                "Pocket park development (1 per 5 blocks)",
                "Green roof incentive program",
                "Community garden network"
            ]
        })
    
    # City-specific additions
    for key, value in city_template["layout_recommendations"].items():
        strategies.append({
            "type": key.replace("_", " ").title(),
            "description": value,
            "specifics": ["Tailored to local conditions", "Phased implementation", "Community engagement"]
        })
    
    return strategies

def generate_infrastructure_plan(city_template, neighborhood_data):
    """Generate comprehensive infrastructure improvement plan"""
    
    infrastructure_plan = {}
    
    # Transportation infrastructure
    avg_traffic = neighborhood_data['traffic_congestion'].mean()
    avg_transit = neighborhood_data['public_transport_access'].mean()
    
    transportation = {
        "priority": "HIGH" if avg_traffic > 60 or avg_transit < 50 else "MEDIUM",
        "projects": []
    }
    
    if avg_traffic > 60:
        transportation["projects"].extend([
            "Smart traffic signal optimization",
            "Congestion pricing implementation",
            "Car-free zones in city center"
        ])
    
    if avg_transit < 50:
        transportation["projects"].extend([
            "Bus rapid transit expansion",
            "Bike-share network deployment",
            "Pedestrian infrastructure upgrades"
        ])
    
    infrastructure_plan["transportation"] = transportation
    
    # Green infrastructure
    avg_greenery = neighborhood_data['greenery_percent'].mean()
    green_infra = {
        "priority": "HIGH" if avg_greenery < 30 else "MEDIUM",
        "projects": [
            "Bioswales for stormwater management",
            "Urban forest expansion program",
            "Green roof and wall incentives",
            "Permeable pavement installation"
        ]
    }
    infrastructure_plan["green_infrastructure"] = green_infra
    
    # Energy infrastructure
    energy = {
        "priority": "HIGH",
        "projects": [
            "Distributed solar panel program",
            "District energy systems",
            "Smart grid implementation",
            "Energy storage facilities"
        ]
    }
    infrastructure_plan["energy"] = energy
    
    # Water infrastructure
    water = {
        "priority": "MEDIUM",
        "projects": [
            "Rainwater harvesting systems",
            "Greywater recycling facilities",
            "Smart water meters",
            "Wetland restoration"
        ]
    }
    infrastructure_plan["water"] = water
    
    return infrastructure_plan

def generate_implementation_phases(city_template, neighborhood_data):
    """Generate phased implementation timeline"""
    
    phases = {
        "Phase 1 (0-2 years)": {
            "focus": "Quick wins and foundation",
            "projects": [
                "Policy framework development",
                "Community engagement programs",
                "Pilot green infrastructure projects",
                "Transportation demand management"
            ],
            "budget_estimate": "Low-Medium",
            "expected_impact": "15-25% improvement in key metrics"
        },
        "Phase 2 (2-5 years)": {
            "focus": "Major infrastructure deployment",
            "projects": [
                "Transit system expansion",
                "Large-scale green space development",
                "Energy system upgrades",
                "Smart city technology integration"
            ],
            "budget_estimate": "High",
            "expected_impact": "40-60% improvement in key metrics"
        },
        "Phase 3 (5-10 years)": {
            "focus": "System optimization and expansion",
            "projects": [
                "Complete green network",
                "Advanced mobility solutions",
                "Carbon neutrality achievement",
                "Resilience infrastructure"
            ],
            "budget_estimate": "Medium-High",
            "expected_impact": "70-90% improvement in key metrics"
        }
    }
    
    return phases

def generate_sustainability_targets(neighborhood_data):
    """Generate specific sustainability targets based on current conditions"""
    
    current_avg_greenery = neighborhood_data['greenery_percent'].mean()
    current_avg_density = neighborhood_data['population_density'].mean()
    current_avg_air = neighborhood_data['air_quality_index'].mean()
    
    targets = {
        "2030_targets": {
            "greenery_coverage": f"{min(current_avg_greenery * 1.5, 80):.1f}%",
            "air_quality_index": f"{max(current_avg_air * 0.7, 25):.0f}",
            "transit_accessibility": "90% within 10-min walk",
            "renewable_energy": "60% of city energy needs",
            "waste_diversion": "80% from landfills"
        },
        "2050_targets": {
            "carbon_neutrality": "Net-zero emissions",
            "greenery_coverage": "50% minimum citywide",
            "circular_economy": "90% material reuse rate",
            "climate_resilience": "100% infrastructure adapted",
            "social_equity": "Equal access to green amenities"
        }
    }
    
    return targets



# Generate suggestions (enhanced version)
def generate_suggestions(row, has_prediction=False):
    """Generate AI-powered actionable suggestions"""
    suggestions = []
    priority = "MEDIUM"
    
    # Critical conditions
    if row['greenery_percent'] < 25 and row['population_density'] > 10000:
        priority = "🔴 CRITICAL"
        suggestions.append("Immediate action required: Urban heat island risk")
        suggestions.append("Deploy vertical gardens and green roofs")
        suggestions.append("Create mandatory green space regulations")
    
    # Greenery analysis
    if row['greenery_percent'] < 30:
        suggestions.append("🌲 Increase tree canopy coverage by 15%")
        suggestions.append("🏞️ Develop pocket parks in vacant lots")
    elif row['greenery_percent'] > 60:
        suggestions.append("🛡️ PRESERVE: Implement conservation zoning")
        suggestions.append("📋 Create environmental protection district")
    
    # Density management
    if row['population_density'] > 10000:
        suggestions.append("🏢 Promote vertical mixed-use development")
        suggestions.append("🚇 Expand metro/bus rapid transit lines")
    
    # Air quality
    if row.get('air_quality_index', 0) > 100:
        suggestions.append("💨 Air quality improvement needed")
        suggestions.append("🚗 Implement low-emission zones")
    
    # Traffic
    if row.get('traffic_congestion', 0) > 70:
        suggestions.append("🚦 Traffic optimization: smart signals needed")
        suggestions.append("🚴 Add protected bike lanes")
    
    # Prediction-based
    if has_prediction and row.get('greenery_change', 0) < -10:
        priority = "🔴 CRITICAL"
        suggestions.append(f"⚠️ ALERT: Projected {abs(row['greenery_change']):.1f}% greenery loss in 5 years")
        suggestions.append("🚨 Immediate conservation measures required")
    
    if not suggestions:
        suggestions.append("✅ Area is well-balanced, maintain current policies")
    
    return suggestions, priority





# Load data
if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("📤 Upload a CSV or use sample urban data")
        df = load_sample_urban_data()
elif data_source == "Extended Urban Data":
    df = load_extended_urban_data()
elif data_source == "Fallback Sample Data":
    df = generate_fallback_data()
else:
    # Default to sample urban data
    df = load_sample_urban_data()

# Calculate scores
df['sustainability_score'] = calculate_sustainability_score(df, greenery_weight, density_weight, infra_weight)

# Apply ML models
if use_clustering:
    df, cluster_labels = perform_clustering(df, n_clusters=3)

if use_prediction:
    df = predict_future_greenery(df)



# Main Content Area - Conditional Page Rendering
if st.session_state.page_selection == "Interactive Map":
    st.header("Interactive Sustainability Map")
    
    col_map, col_metrics = st.columns([2.5, 1])
    
    with col_map:
        # Create map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for idx, row in df.iterrows():
            score = row['sustainability_score']
            
            # Color coding
            if score >= 70:
                color, icon = 'green', 'leaf'
            elif score >= 50:
                color, icon = 'lightgreen', 'tree'
            elif score >= 30:
                color, icon = 'orange', 'exclamation-triangle'
            else:
                color, icon = 'red', 'exclamation-circle'
            
            # Popup with detailed info
            city_info = f"<b>🏙️ City:</b> {row['city']}<br>" if 'city' in df.columns else ""
            popup_html = f"""
            <div style="width: 280px; font-family: Arial;">
                <h4 style="margin-bottom: 10px;">{row['neighborhood']}</h4>
                {city_info}
                <b>🎯 Sustainability Score:</b> {score:.1f}/100<br>
                <b>🌳 Greenery:</b> {row['greenery_percent']:.1f}%<br>
                <b>👥 Pop Density:</b> {row['population_density']:.0f}/km²<br>
                <b>🏗️ Infrastructure:</b> {row['infrastructure_score']:.1f}/100<br>
                <b>💨 Air Quality:</b> {row['air_quality_index']:.0f} AQI<br>
                {'<b>🔮 5Y Prediction:</b> ' + f"{row['predicted_greenery_5y']:.1f}%" if use_prediction else ''}
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['neighborhood']}: {score:.1f}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Add heatmap layer
        if len(df) > 0:
            heat_data = [[row['latitude'], row['longitude'], row['sustainability_score']] 
                        for idx, row in df.iterrows()]
            plugins.HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
        
        st_folium(m, width=800, height=600, returned_objects=[])
    
    with col_metrics:
        st.subheader("📊 City Overview")
        
        avg_score = df['sustainability_score'].mean()
        avg_green = df['greenery_percent'].mean()
        critical_areas = len(df[df['sustainability_score'] < 30])
        
        st.metric("🎯 Avg Sustainability", f"{avg_score:.1f}/100", 
                 delta=f"{avg_score - 50:.1f} vs baseline")
        st.metric("🌳 Avg Green Coverage", f"{avg_green:.1f}%")
        st.metric("⚠️ Critical Areas", critical_areas, 
                 delta_color="inverse")
        
        if use_prediction:
            avg_future = df['predicted_greenery_5y'].mean()
            change = avg_future - avg_green
            st.metric("🔮 Predicted Green (5Y)", f"{avg_future:.1f}%",
                     delta=f"{change:+.1f}%", delta_color="normal" if change >= 0 else "inverse")
        
        st.markdown("---")
        
        # Data source info
        st.markdown("**📊 Data Source:**")
        if data_source == "Sample Urban Data":
            st.success("🗃️ Sample Urban Dataset")
            st.markdown(f"*{len(df)} neighborhoods analyzed*")
        elif data_source == "Extended Urban Data":
            st.success("🗃️ Extended Urban Dataset")
            st.markdown(f"*{len(df)} neighborhoods analyzed*")
        else:
            st.info(f"📁 {data_source}")
        
        st.markdown("---")
        
        # Quick stats
        best_area = df.loc[df['sustainability_score'].idxmax(), 'neighborhood']
        worst_area = df.loc[df['sustainability_score'].idxmin(), 'neighborhood']
        
        st.markdown(f"**🏆 Best:** {best_area}")
        st.markdown(f"**⚠️ Needs Help:** {worst_area}")

elif st.session_state.page_selection == "ML Insights":
    st.header("🤖 Machine Learning Insights")
    
    if use_clustering:
        st.subheader("K-Means Clustering Analysis")
        st.markdown("Neighborhoods grouped by sustainability characteristics:")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # 3D scatter plot
            fig_3d = px.scatter_3d(
                df,
                x='greenery_percent',
                y='population_density',
                z='infrastructure_score',
                color='cluster_label',
                hover_name='neighborhood',
                title="3D Cluster Visualization",
                labels={
                    'greenery_percent': 'Greenery %',
                    'population_density': 'Pop Density',
                    'infrastructure_score': 'Infrastructure'
                },
                color_discrete_map={
                    "🌳 Green & Sustainable": "#00C853",
                    "🏙️ High Density Urban": "#FF6F00",
                    "⚠️ Needs Improvement": "#D32F2F"
                }
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.markdown("**Cluster Distribution:**")
            cluster_counts = df['cluster_label'].value_counts()
            
            for label, count in cluster_counts.items():
                st.markdown(f"**{label}:** {count} neighborhoods")
            
            st.markdown("---")
            st.markdown("**Cluster Characteristics:**")
            for i, label in enumerate(cluster_labels):
                mask = df['cluster'] == i
                avg_score = df[mask]['sustainability_score'].mean()
                st.markdown(f"{label}: Avg Score = {avg_score:.1f}")
    
    if use_prediction:
        st.subheader("🔮 Predictive Analysis: 5-Year Greenery Forecast")
        
        # Prediction comparison chart
        pred_df = df[['neighborhood', 'greenery_percent', 'predicted_greenery_5y', 'greenery_change']].copy()
        pred_df = pred_df.sort_values('greenery_change')
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Bar(
            name='Current Greenery',
            x=pred_df['neighborhood'],
            y=pred_df['greenery_percent'],
            marker_color='lightgreen'
        ))
        
        fig_pred.add_trace(go.Bar(
            name='Predicted (5Y)',
            x=pred_df['neighborhood'],
            y=pred_df['predicted_greenery_5y'],
            marker_color='darkgreen'
        ))
        
        fig_pred.update_layout(
            title="Current vs Predicted Greenery Coverage",
            xaxis_title="Neighborhood",
            yaxis_title="Greenery %",
            barmode='group',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Alert for critical predictions
        critical_predictions = df[df['greenery_change'] < -10]
        if len(critical_predictions) > 0:
            st.error(f"🚨 **ALERT:** {len(critical_predictions)} neighborhoods face critical greenery loss!")
            for idx, row in critical_predictions.iterrows():
                st.warning(f"**{row['neighborhood']}**: Projected {row['greenery_change']:.1f}% loss")

elif st.session_state.page_selection == "AI Recommendations":
    st.header("💡 AI-Generated Recommendations")
    
    # Priority filter
    priority_filter = st.multiselect(
        "Filter by priority:",
        ["🔴 CRITICAL", "MEDIUM"],
        default=["🔴 CRITICAL", "MEDIUM"]
    )
    
    for idx, row in df.sort_values('sustainability_score').iterrows():
        suggestions, priority = generate_suggestions(row, use_prediction)
        
        if priority in priority_filter or ("MEDIUM" in priority_filter and priority == "MEDIUM"):
            with st.expander(f"**{row['neighborhood']}** | Score: {row['sustainability_score']:.1f} | {priority}"):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.metric("🌳 Greenery", f"{row['greenery_percent']:.1f}%")
                    st.metric("👥 Density", f"{row['population_density']:.0f}")
                    st.metric("🏗️ Infrastructure", f"{row['infrastructure_score']:.1f}")
                    
                    if use_clustering:
                        st.markdown(f"**Cluster:** {row['cluster_label']}")
                
                with col_b:
                    st.markdown("**🎯 Recommended Actions:**")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.markdown(f"{i}. {suggestion}")

elif st.session_state.page_selection == "City Layouts & Infrastructure":
    st.header("🏗️ Sustainable City Layouts & Infrastructure")
    
    # Get city-specific recommendations
    if 'city' in df.columns and len(df['city'].unique()) > 0:
        selected_city = st.selectbox(
            "🏙️ Select City for Detailed Planning:",
            options=df['city'].unique(),
            index=0
        )
        
        # Filter data for selected city
        city_data = df[df['city'] == selected_city]
        
        # Generate city-specific recommendations
        recommendations, city_template = get_city_specific_recommendations(selected_city, city_data)
        
        # City Overview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📍 {selected_city} Overview")
            st.markdown(f"**Climate Type:** {city_template['climate'].replace('_', ' ').title()}")
            
            st.markdown("**Key Challenges:**")
            for challenge in city_template['challenges']:
                st.markdown(f"• {challenge.replace('_', ' ').title()}")
            
            st.markdown("**City Strengths:**")
            for strength in city_template['strengths']:
                st.markdown(f"• {strength.replace('_', ' ').title()}")
        
        with col2:
            st.subheader("📊 Current City Metrics")
            avg_sustainability = city_data['sustainability_score'].mean()
            avg_greenery = city_data['greenery_percent'].mean()
            avg_density = city_data['population_density'].mean()
            neighborhoods_count = len(city_data)
            
            st.metric("🎯 Avg Sustainability Score", f"{avg_sustainability:.1f}/100")
            st.metric("🌳 Avg Green Coverage", f"{avg_greenery:.1f}%")
            st.metric("👥 Avg Population Density", f"{avg_density:.0f}/km²")
            st.metric("🏘️ Neighborhoods Analyzed", neighborhoods_count)
        
        st.markdown("---")
        
        # Layout Strategy Section
        st.subheader("🗺️ Sustainable Layout Strategy")
        
        layout_strategies = recommendations["layout_strategy"]
        
        for i, strategy in enumerate(layout_strategies):
            with st.expander(f"**Strategy {i+1}: {strategy['type']}**"):
                st.markdown(f"**Description:** {strategy['description']}")
                st.markdown("**Specific Actions:**")
                for specific in strategy['specifics']:
                    st.markdown(f"• {specific}")
        
        # Infrastructure Plan Section
        st.subheader("🏗️ Infrastructure Improvement Plan")
        
        infra_plan = recommendations["infrastructure_plan"]
        
        # Create infrastructure priority visualization
        infra_priorities = []
        for category, data in infra_plan.items():
            infra_priorities.append({
                "Category": category.replace("_", " ").title(),
                "Priority": data["priority"],
                "Projects": len(data["projects"])
            })
        
        priority_df = pd.DataFrame(infra_priorities)
        
        # Priority chart
        fig_priority = px.bar(
            priority_df,
            x="Category",
            y="Projects",
            color="Priority",
            title="Infrastructure Investment Priorities",
            color_discrete_map={"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#44AA44"}
        )
        st.plotly_chart(fig_priority, width='stretch')
        
        infra_col1, infra_col2 = st.columns(2)
        
        with infra_col1:
            # Transportation
            transport = infra_plan["transportation"]
            priority_color = "🔴" if transport["priority"] == "HIGH" else "🟡"
            with st.container():
                st.markdown(f"### 🚇 Transportation {priority_color}")
                st.markdown(f"**Priority Level:** {transport['priority']}")
                for project in transport["projects"]:
                    st.markdown(f"• {project}")
            
            st.markdown("---")
            
            # Green Infrastructure
            green = infra_plan["green_infrastructure"]
            priority_color = "🔴" if green["priority"] == "HIGH" else "🟡"
            with st.container():
                st.markdown(f"### 🌳 Green Infrastructure {priority_color}")
                st.markdown(f"**Priority Level:** {green['priority']}")
                for project in green["projects"]:
                    st.markdown(f"• {project}")
        
        with infra_col2:
            # Energy
            energy = infra_plan["energy"]
            priority_color = "🔴" if energy["priority"] == "HIGH" else "🟡"
            with st.container():
                st.markdown(f"### ⚡ Energy Systems {priority_color}")
                st.markdown(f"**Priority Level:** {energy['priority']}")
                for project in energy["projects"]:
                    st.markdown(f"• {project}")
            
            st.markdown("---")
            
            # Water
            water = infra_plan["water"]
            priority_color = "🔴" if water["priority"] == "HIGH" else "🟡"
            with st.container():
                st.markdown(f"### 💧 Water Management {priority_color}")
                st.markdown(f"**Priority Level:** {water['priority']}")
                for project in water["projects"]:
                    st.markdown(f"• {project}")
        
        st.markdown("---")
        
        # Implementation Timeline
        st.subheader("📅 Implementation Timeline")
        
        phases = recommendations["implementation_phases"]
        
        timeline_cols = st.columns(3)
        
        for i, (phase_name, phase_data) in enumerate(phases.items()):
            with timeline_cols[i]:
                st.markdown(f"**{phase_name}**")
                st.markdown(f"*Focus: {phase_data['focus']}*")
                
                st.markdown("**Key Projects:**")
                for project in phase_data["projects"]:
                    st.markdown(f"• {project}")
                
                st.markdown(f"**Budget:** {phase_data['budget_estimate']}")
                st.markdown(f"**Impact:** {phase_data['expected_impact']}")
        
        st.markdown("---")
        
        # Sustainability Targets
        st.subheader("🎯 Sustainability Targets")
        
        targets = recommendations["sustainability_metrics"]
        
        target_col1, target_col2 = st.columns(2)
        
        with target_col1:
            st.markdown("**🗓️ 2030 Targets**")
            for key, value in targets["2030_targets"].items():
                st.markdown(f"• **{key.replace('_', ' ').title()}:** {value}")
        
        with target_col2:
            st.markdown("**🗓️ 2050 Vision**")
            for key, value in targets["2050_targets"].items():
                st.markdown(f"• **{key.replace('_', ' ').title()}:** {value}")
        
        # Interactive Layout Visualization
        st.markdown("---")
        st.subheader("🗺️ Proposed Layout Visualization")
        
        # Create a conceptual map showing proposed improvements
        center_lat = city_data['latitude'].mean()
        center_lon = city_data['longitude'].mean()
        
        layout_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add existing neighborhoods
        for idx, row in city_data.iterrows():
            # Color based on improvement priority
            if row['sustainability_score'] < 40:
                color = 'red'
                icon = 'exclamation-triangle'
                improvement = "High Priority Redevelopment"
            elif row['sustainability_score'] < 60:
                color = 'orange'
                icon = 'wrench'
                improvement = "Moderate Improvements Needed"
            else:
                color = 'green'
                icon = 'leaf'
                improvement = "Maintain & Enhance"
            
            popup_html = f"""
            <div style="width: 300px; font-family: Arial;">
                <h4>{row['neighborhood']}</h4>
                <b>Current Score:</b> {row['sustainability_score']:.1f}/100<br>
                <b>Improvement Strategy:</b> {improvement}<br>
                <b>Priority Actions:</b><br>
                • Green infrastructure expansion<br>
                • Transit connectivity<br>
                • Mixed-use development<br>
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row['neighborhood']}: {improvement}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(layout_map)
        
        # Add proposed infrastructure (conceptual)
        # Green corridors
        if len(city_data) > 1:
            # Create a conceptual green corridor
            coords = [[row['latitude'], row['longitude']] for _, row in city_data.iterrows()]
            folium.PolyLine(
                coords,
                color='green',
                weight=6,
                opacity=0.7,
                popup="Proposed Green Corridor Network"
            ).add_to(layout_map)
        
        st_folium(layout_map, width=800, height=500, returned_objects=[])
        
        # Download recommendations
        st.markdown("---")
        
        # Create downloadable report
        report_data = {
            "City": selected_city,
            "Current_Avg_Sustainability": avg_sustainability,
            "Current_Avg_Greenery": avg_greenery,
            "Neighborhoods_Count": neighborhoods_count,
            "Layout_Strategies": len(layout_strategies),
            "Infrastructure_Projects": sum(len(plan["projects"]) for plan in infra_plan.values()),
            "Implementation_Phases": len(phases)
        }
        
        report_df = pd.DataFrame([report_data])
        csv_report = report_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download City Planning Report",
            data=csv_report,
            file_name=f"{selected_city.replace(' ', '_')}_sustainable_plan.csv",
            mime="text/csv"
        )
        
    else:
        st.info("🏙️ City-specific planning requires data with city information. Using general recommendations.")
        
        # General recommendations for all neighborhoods
        st.subheader("🌍 General Sustainable Urban Planning Principles")
        
        general_principles = [
            {
                "principle": "15-Minute City Concept",
                "description": "Ensure all residents can access daily needs within a 15-minute walk or bike ride",
                "implementation": ["Mixed-use zoning", "Distributed services", "Active transportation"]
            },
            {
                "principle": "Green-Blue Infrastructure",
                "description": "Integrate natural systems for multiple urban functions",
                "implementation": ["Urban forests", "Wetlands", "Green roofs", "Bioswales"]
            },
            {
                "principle": "Transit-Oriented Development",
                "description": "Concentrate development around public transit nodes",
                "implementation": ["High-density housing", "Commercial centers", "Bike facilities"]
            },
            {
                "principle": "Climate Resilience",
                "description": "Design infrastructure to adapt to climate change",
                "implementation": ["Flood management", "Heat mitigation", "Energy efficiency"]
            }
        ]
        
        for principle in general_principles:
            with st.expander(f"**{principle['principle']}**"):
                st.markdown(f"**Description:** {principle['description']}")
                st.markdown("**Implementation Strategies:**")
                for impl in principle['implementation']:
                    st.markdown(f"• {impl}")

elif st.session_state.page_selection == "Analytics":
    st.header("📊 Advanced Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        corr_features = ['greenery_percent', 'population_density', 'infrastructure_score', 
                        'air_quality_index', 'sustainability_score']
        corr_matrix = df[corr_features].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdYlGn',
            title="Feature Correlations"
        )
        st.plotly_chart(fig_corr, width='stretch')
    
    with chart_col2:
        # Scatter matrix
        st.subheader("Multi-variable Analysis")
        fig_scatter = px.scatter(
            df,
            x='air_quality_index',
            y='greenery_percent',
            size='population_density',
            color='sustainability_score',
            hover_name='neighborhood',
            title="Air Quality vs Greenery",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    # Data table
    st.subheader("📋 Complete Dataset")
    display_cols = ['neighborhood']
    
    # Add city column if it exists
    if 'city' in df.columns:
        display_cols.append('city')
    
    display_cols.extend(['sustainability_score', 'greenery_percent', 
                        'population_density', 'infrastructure_score'])
    
    if use_clustering:
        display_cols.append('cluster_label')
    if use_prediction:
        display_cols.extend(['predicted_greenery_5y', 'greenery_change'])
    
    display_df = df[display_cols].round(2)
    display_df = display_df.sort_values('sustainability_score', ascending=False)
    
    st.dataframe(
        display_df.style.background_gradient(subset=['sustainability_score'], cmap='RdYlGn'),
        width='stretch',
        height=400
    )
    
    # Download
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download Analysis Results",
        data=csv,
        file_name="urban_ai_analysis.csv",
        mime="text/csv"
    )





# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p style="color: #666;"><b>UPlanning</b> - Smart Urban Development Platform</p>
    <p style="color: #999; font-size: 0.9em;">AI Analytics | Sustainability Focus | Predictive Modeling | Smart City Planning</p>
</div>
""", unsafe_allow_html=True)