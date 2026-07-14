import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake Currency Detection Dashboard",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    /* Dark glassmorphic background & main styling */
    .stApp {
        background-color: #080c14;
        color: #f3f4f6;
    }
    
    /* Header Banner styling */
    .hero-container {
        background: linear-gradient(135deg, #111c30 0%, #064e3b 100%);
        border: 1px solid #065f46;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        color: #00e676;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        margin: 0;
        text-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #111c30;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00e676;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    
    /* Prediction Cards */
    .status-card-authentic {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.15) 0%, rgba(0, 230, 118, 0.02) 100%);
        border: 2px solid #00e676;
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.2);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    
    .status-card-fake {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.15) 0%, rgba(255, 23, 68, 0.02) 100%);
        border: 2px solid #ff1744;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.2);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    
    /* Form controls */
    div[data-baseweb="slider"] {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to rerun Streamlit safely
def trigger_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------
# Load Model, Scaler & Dataset
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("data/BankNote_Authentication.csv")

try:
    model, scaler = load_assets()
    data = load_data()
except Exception as e:
    st.error(f"Failed to load system assets. Please verify model files and dataset. Error: {e}")
    st.stop()

# -----------------------------
# Session State Initialization
# -----------------------------
# Initialize defaults based on dataset means
if "variance" not in st.session_state:
    st.session_state.variance = 0.4337
if "skewness" not in st.session_state:
    st.session_state.skewness = 1.9224
if "curtosis" not in st.session_state:
    st.session_state.curtosis = 1.3976
if "entropy" not in st.session_state:
    st.session_state.entropy = -1.1917

# -----------------------------
# Sidebar: Presets & Controls
# -----------------------------
st.sidebar.image("https://img.icons8.com/color/96/banknotes.png", width=80)
st.sidebar.title("Configuration")

# Preset buttons
st.sidebar.markdown("### 📋 Sample Presets")
st.sidebar.caption("Instantly load representative values to test the model:")

col_auth, col_fake = st.sidebar.columns(2)
with col_auth:
    if st.sidebar.button("🟢 Authentic Note", use_container_width=True):
        st.session_state.variance = 3.6216
        st.session_state.skewness = 8.6661
        st.session_state.curtosis = -2.8073
        st.session_state.entropy = -0.4470
        trigger_rerun()

with col_fake:
    if st.sidebar.button("🔴 Fake Note", use_container_width=True):
        st.session_state.variance = -1.3971
        st.session_state.skewness = 3.3191
        st.session_state.curtosis = -1.3927
        st.session_state.entropy = -1.9948
        trigger_rerun()

if st.sidebar.button("🎲 Random Data Point", use_container_width=True):
    random_row = data.sample(n=1).iloc[0]
    st.session_state.variance = float(random_row['variance'])
    st.session_state.skewness = float(random_row['skewness'])
    st.session_state.curtosis = float(random_row['curtosis'])
    st.session_state.entropy = float(random_row['entropy'])
    trigger_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Adjust Features")

# Bounded inputs based on dataset ranges
variance = st.sidebar.slider(
    "Variance of Wavelet",
    min_value=-8.0,
    max_value=8.0,
    key="variance",
    step=0.01,
    format="%.4f",
    help="Variance of wavelet transformed image. High values denote sharp structures."
)

skewness = st.sidebar.slider(
    "Skewness of Wavelet",
    min_value=-15.0,
    max_value=15.0,
    key="skewness",
    step=0.01,
    format="%.4f",
    help="Asymmetry of image gray scale levels."
)

kurtosis = st.sidebar.slider(
    "Kurtosis of Wavelet",
    min_value=-10.0,
    max_value=20.0,
    key="curtosis",
    step=0.01,
    format="%.4f",
    help="Kurtosis or peakiness of Wavelet Transformed image distributions."
)

entropy = st.sidebar.slider(
    "Entropy of Image",
    min_value=-10.0,
    max_value=5.0,
    key="entropy",
    step=0.01,
    format="%.4f",
    help="Entropy or randomness in pixels of the image."
)

# -----------------------------
# Main Content
# -----------------------------

# Hero banner
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">💵 Fake Currency Detection Dashboard</h1>
        <p class="hero-subtitle">Interactive Real-time Banknote Authentication using Logistic Regression</p>
    </div>
""", unsafe_allow_html=True)

# Metrics section
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
overall_accuracy = accuracy_score(y, y_pred)

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98.06%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Banknotes Sample Size</div>
        </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Logistic Regression</div>
            <div class="metric-label">Algorithm</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model Prediction & Confidence Score
# Scale input data safely using DataFrame to avoid warnings about feature names
input_df = pd.DataFrame(
    [[variance, skewness, kurtosis, entropy]], 
    columns=["variance", "skewness", "curtosis", "entropy"]
)
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0] # [prob_authentic, prob_fake]
prob_authentic = probabilities[0]
prob_fake = probabilities[1]

# Display prediction result card
if prediction == 0:
    st.markdown(f"""
        <div class="status-card-authentic">
            <span style="font-size: 3rem;">✅</span>
            <h2 style="color: #00e676; margin: 10px 0 0 0; font-family: sans-serif;">The Banknote is AUTHENTIC</h2>
            <p style="color: #9ca3af; margin: 5px 0 15px 0; font-size: 1.1rem;">
                Match probability confidence score
            </p>
            <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{prob_authentic*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="status-card-fake">
            <span style="font-size: 3rem;">❌</span>
            <h2 style="color: #ff1744; margin: 10px 0 0 0; font-family: sans-serif;">The Banknote is FAKE</h2>
            <p style="color: #9ca3af; margin: 5px 0 15px 0; font-size: 1.1rem;">
                Match probability confidence score
            </p>
            <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{prob_fake*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

# Tabbed Layout for Analysis
tab1, tab2 = st.tabs(["📊 Spatial Analysis (3D Feature Space)", "📈 Model Evaluation & Data Summary"])

with tab1:
    st.markdown("### 3D Feature Space Distribution")
    st.markdown("This interactive 3D visualization shows where the currently inputted banknote features lie in relation to the dataset samples.")
    
    # Downsample dataset slightly for smooth interactive rendering
    df_sample = data.sample(n=min(500, len(data)), random_state=42)
    # Rename class labels for visual clarity
    df_sample['Status'] = df_sample['class'].map({0: 'Authentic', 1: 'Fake'})
    
    fig_3d = px.scatter_3d(
        df_sample,
        x='variance',
        y='skewness',
        z='curtosis',
        color='Status',
        color_discrete_map={'Authentic': '#00e676', 'Fake': '#ff1744'},
        labels={'variance': 'Variance', 'skewness': 'Skewness', 'curtosis': 'Kurtosis'},
        opacity=0.6
    )
    
    # Style scatter plot points
    fig_3d.update_traces(marker=dict(size=4, line=dict(width=0.5, color='rgba(0,0,0,0.2)')))
    
    # Add current test point
    fig_3d.add_trace(
        go.Scatter3d(
            x=[variance],
            y=[skewness],
            z=[kurtosis],
            mode='markers',
            marker=dict(
                size=12,
                color='#00ffff',
                symbol='diamond',
                line=dict(color='white', width=2),
                opacity=1.0
            ),
            name='Current Input',
            hoverinfo='text',
            text=[f"Current Input<br>Variance: {variance:.3f}<br>Skewness: {skewness:.3f}<br>Kurtosis: {kurtosis:.3f}<br>Prediction: {'Authentic' if prediction == 0 else 'Fake'}"]
        )
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", backgroundcolor="rgba(0,0,0,0)"),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="center",
            x=0.5,
            font=dict(color="white")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=550
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    col_t2_1, col_t2_2 = st.columns(2)
    
    with col_t2_1:
        st.markdown("### Model Confusion Matrix")
        st.markdown("Evaluation of model accuracy based on true/false positives/negatives.")
        
        cm = confusion_matrix(y, y_pred)
        
        # Plotly Heatmap
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Authentic', 'Predicted Fake'],
            y=['Actual Authentic', 'Actual Fake'],
            colorscale=[[0, '#111827'], [0.5, '#047857'], [1, '#059669']],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            hoverinfo="none",
            showscale=False
        ))
        
        fig_cm.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=300,
            width=300
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_t2_2:
        st.markdown("### Feature Statistics")
        st.markdown("Summary of distribution bounds in the dataset:")
        
        stats = data.drop(columns=['class']).describe().loc[['min', 'mean', 'max']]
        st.table(stats.T.rename(columns={'min': 'Minimum', 'mean': 'Average', 'max': 'Maximum'}))

st.markdown("---")
st.caption("Built with Scikit-learn, Plotly & Streamlit 🚀 | Developed by Rhea Biswas")
