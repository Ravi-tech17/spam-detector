import streamlit as st
import pickle
import os

# Page config
st.set_page_config(
    page_title="🚀 Spam Detector Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .result-spam {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(255,107,107,0.4);
    }
    .result-ham {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(81,207,102,0.4);
    }
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    if not all(os.path.exists(f) for f in ["vectorizer.pkl", "model.pkl"]):
        st.error("❌ **Run `python train.py` first!**")
        st.stop()
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model


tfidf, model = load_models()

# Header
st.markdown('<h1 class="main-header">🚀 Spam Detector Pro</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", "5,568")
    with col2:
        st.metric("Accuracy", "96.5%")

    st.markdown("---")
    st.markdown("### 🧪 Quick Tests")
    test_messages = {
        "🚫 Spam Test": "FREE entry in 2 a wkly comp to win FA Cup final tkts",
        "✅ Ham Test": "Hey, how are you doing today?",
        "🎁 Prize Spam": "WINNER!! As a valued network customer you have been selected"
    }

    selected_test = st.selectbox("Quick Test:", list(test_messages.keys()))
    if st.button("🔥 Test Selected", type="secondary"):
        st.session_state.test_message = test_messages[selected_test]

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Enter Message")
    message = st.text_area(
        "",
        height=200,
        placeholder="Type your SMS/Email here...",
        key="message_input"
    )

    # Auto-fill test message
    if "test_message" in st.session_state:
        message = st.session_state.test_message
        st.session_state.test_message = None

with col2:
    st.markdown("### 🎯 Examples")
    st.info("**Spam:** FREE, WINNER, £900 prize")
    st.info("**Not Spam:** Hey, meeting, thanks")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 **ANALYZE MESSAGE**", type="primary", use_container_width=True):
        if message.strip():
            with st.spinner("🔬 Analyzing..."):
                vectorized = tfidf.transform([message])
                result = model.predict(vectorized)[0]
                prob = model.predict_proba(vectorized)[0]

                # Clear previous results
                st.session_state.clear()

                # Show result
                if result == 1:
                    st.markdown(f"""
                    <div class="result-spam">
                        🚫 **SPAM DETECTED!**<br>
                        <small>Spam Confidence: <strong>{prob[1]:.1%}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-ham">
                        ✅ **SAFE MESSAGE**<br> 
                        <small>Ham Confidence: <strong>{prob[0]:.1%}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)

                # Show confidence bar
                st.progress(max(prob))

        else:
            st.warning("⚠️ **Please enter a message!**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using <strong>Streamlit</strong> + <strong>scikit-learn</strong> 
    | Accuracy: <strong>96.5%</strong>
</div>
""", unsafe_allow_html=True)