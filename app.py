import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PAGE CONFIG + BRAND (matches the workshop slide deck: 3428BA / 5284FF)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic AI Challenge — Live Prediction",
    page_icon="🚢",
    layout="wide",
)

PRIMARY = "#3428BA"
ACCENT = "#5284FF"
DARK = "#16123A"
CARD = "#F4F3FC"
MUTED = "#6E6B8F"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: #FFFFFF;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    .hero {{
        background: linear-gradient(135deg, {DARK} 0%, {PRIMARY} 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
    }}
    .hero-eyebrow {{
        color: {ACCENT};
        font-weight: 700;
        letter-spacing: 3px;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }}
    .hero-title {{
        color: white;
        font-weight: 800;
        font-size: 2.6rem;
        line-height: 1.15;
        margin: 0;
    }}
    .hero-sub {{
        color: #C9C6E8;
        font-size: 1.05rem;
        font-style: italic;
        margin-top: 0.6rem;
    }}

    .card {{
        background-color: {CARD};
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }}
    .card-dark {{
        background-color: {PRIMARY};
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        color: white;
    }}
    .section-label {{
        color: {ACCENT};
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 0.78rem;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .big-stat {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {PRIMARY};
        margin: 0;
    }}
    .big-stat-label {{
        color: {MUTED};
        font-size: 0.85rem;
        margin-top: -0.3rem;
    }}

    .result-survive {{
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
    }}
    .result-perish {{
        background: linear-gradient(135deg, #7A1F1F 0%, #B23A3A 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
    }}
    .result-title {{
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }}
    .result-conf {{
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }}

    div.stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
    }}
    div.stButton > button:hover {{
        background-color: {ACCENT};
        color: white;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

bundle = load_model()
model = bundle["model"]
features = bundle["features"]
test_accuracy = bundle["test_accuracy"]
cv_mean = bundle["cv_mean"]
cv_std = bundle["cv_std"]
importances = bundle["feature_importances"]

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">TITANIC AI CHALLENGE — LIVE DEMO</div>
    <p class="hero-title">Would You Have Survived?</p>
    <p class="hero-sub">Build a passenger below and watch a real, trained AI model predict their fate — live, no coding involved.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL STATS ROW
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
stats = [
    (f"{test_accuracy*100:.0f}%", "Test Accuracy"),
    (f"{cv_mean*100:.0f}%", "Cross-Val Accuracy"),
    ("891", "Passengers It Learned From"),
    ("Random Forest", "The Algorithm"),
]
for col, (val, label) in zip([c1, c2, c3, c4], stats):
    col.markdown(f"""
    <div class="card" style="text-align:center;">
        <p class="big-stat">{val}</p>
        <p class="big-stat-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# LAYOUT: LEFT = INPUT FORM, RIGHT = RESULT
# ---------------------------------------------------------------------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="section-label">Build a Passenger</div>', unsafe_allow_html=True)

    # ---- Quick presets: the exact 4 passengers used in the workshop opener ----
    st.write("**Quick-load the opening-game passengers:**")
    preset_cols = st.columns(4)
    presets = {
        "22yo Man, 3rd Class": dict(Pclass=3, Sex="male", Age=22, SibSp=0, Parch=0, Fare=8.0, Embarked="S"),
        "4yo Girl, 1st Class": dict(Pclass=1, Sex="female", Age=4, SibSp=1, Parch=2, Fare=81.0, Embarked="S"),
        "34yo Woman Alone, 2nd": dict(Pclass=2, Sex="female", Age=34, SibSp=0, Parch=0, Fare=13.0, Embarked="S"),
        "60yo Man, 1st Class": dict(Pclass=1, Sex="male", Age=60, SibSp=0, Parch=0, Fare=76.0, Embarked="C"),
    }
    if "preset" not in st.session_state:
        st.session_state.preset = None

    for col, name in zip(preset_cols, presets.keys()):
        if col.button(name.split(",")[0], use_container_width=True, key=f"btn_{name}"):
            st.session_state.preset = name

    defaults = presets.get(st.session_state.preset, dict(
        Pclass=3, Sex="male", Age=30, SibSp=0, Parch=0, Fare=15.0, Embarked="S"
    ))

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        pclass = st.selectbox("Ticket Class", [1, 2, 3],
                               index=[1, 2, 3].index(defaults["Pclass"]),
                               format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
        sex = st.radio("Gender", ["male", "female"],
                        index=["male", "female"].index(defaults["Sex"]), horizontal=True)
        age = st.slider("Age", 0, 80, defaults["Age"])
        fare = st.slider("Fare Paid ($)", 0.0, 250.0, float(defaults["Fare"]), step=1.0)
    with fcol2:
        sibsp = st.number_input("Siblings / Spouse Aboard", 0, 8, defaults["SibSp"])
        parch = st.number_input("Parents / Children Aboard", 0, 6, defaults["Parch"])
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                                 index=["S", "C", "Q"].index(defaults["Embarked"]),
                                 format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔮 PREDICT SURVIVAL", use_container_width=True)

with right:
    st.markdown('<div class="section-label">Live Prediction</div>', unsafe_allow_html=True)

    sex_enc = 0 if sex == "male" else 1
    emb_enc = {"S": 0, "C": 1, "Q": 2}[embarked]
    row = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare, emb_enc]], columns=features)

    proba = model.predict_proba(row)[0]
    survive_prob = proba[1]
    pred = int(survive_prob >= 0.5)

    if pred == 1:
        st.markdown(f"""
        <div class="result-survive">
            <p class="result-title">✅ SURVIVES</p>
            <p class="result-conf">{survive_prob*100:.0f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-perish">
            <p class="result-title">⚠️ DOES NOT SURVIVE</p>
            <p class="result-conf">{(1-survive_prob)*100:.0f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.progress(float(survive_prob), text=f"Survival probability: {survive_prob*100:.0f}%")

    st.write("")
    st.markdown('<div class="section-label">What Drove This Prediction?</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        "Feature": ["Gender", "Fare", "Age", "Ticket Class", "Siblings/Spouse", "Parents/Children", "Port"],
        "Importance": [
            importances["Sex_enc"], importances["Fare"], importances["Age"],
            importances["Pclass"], importances["SibSp"], importances["Parch"],
            importances["Embarked_enc"],
        ],
    }).sort_values("Importance", ascending=True)
    st.bar_chart(imp_df.set_index("Feature"), horizontal=True, color=PRIMARY)

    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem;">
        <p style="color:{MUTED}; font-size:0.85rem; margin:0;">
        The model learned this from 891 real passengers — it's spotting the same pattern
        the room spotted by instinct at the start of the workshop: <b>gender, fare/class, and age</b>
        mattered most for who survived.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.caption("Built for the Titanic AI Challenge workshop · Random Forest trained on the classic Titanic dataset (891 passengers) · No coding was used to build this passenger — just the form above.")
