import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title='Bank Deposit Prediction Dashboard',
    page_icon='🏦',
    layout='wide'
)

# ── Warna per tab ─────────────────────────────────────────────
st.markdown("""
<style>
/* Font global */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #a8d8ea !important; }

/* Tombol predict */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #e94560, #0f3460);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 16px;
    padding: 10px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 12px;
    border-left: 5px solid #0f3460;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f0f2f6;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 14px;
}

/* Warna aktif tiap tab */
.stTabs [aria-selected="true"]:nth-child(1) { background: #1a73e8 !important; color: white !important; }
.stTabs [aria-selected="true"]:nth-child(2) { background: #34a853 !important; color: white !important; }
.stTabs [aria-selected="true"]:nth-child(3) { background: #fa7b17 !important; color: white !important; }
.stTabs [aria-selected="true"]:nth-child(4) { background: #9c27b0 !important; color: white !important; }
.stTabs [aria-selected="true"]:nth-child(5) { background: #e53935 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    return model

model = load_model()

# ── Load dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("bank (data final project).csv")

df = load_data()

# ── Kolom fitur (51 hasil OHE) ────────────────────────────────
FEATURE_COLS = [
    'num__age','num__balance','num__day','num__duration','num__campaign',
    'num__pdays','num__previous',
    'cat__job_admin.','cat__job_blue-collar','cat__job_entrepreneur',
    'cat__job_housemaid','cat__job_management','cat__job_retired',
    'cat__job_self-employed','cat__job_services','cat__job_student',
    'cat__job_technician','cat__job_unemployed','cat__job_unknown',
    'cat__marital_divorced','cat__marital_married','cat__marital_single',
    'cat__education_primary','cat__education_secondary',
    'cat__education_tertiary','cat__education_unknown',
    'cat__default_no','cat__default_yes',
    'cat__housing_no','cat__housing_yes',
    'cat__loan_no','cat__loan_yes',
    'cat__contact_cellular','cat__contact_telephone','cat__contact_unknown',
    'cat__month_apr','cat__month_aug','cat__month_dec','cat__month_feb',
    'cat__month_jan','cat__month_jul','cat__month_jun','cat__month_mar',
    'cat__month_may','cat__month_nov','cat__month_oct','cat__month_sep',
    'cat__poutcome_failure','cat__poutcome_other',
    'cat__poutcome_success','cat__poutcome_unknown'
]

def build_input(age, balance, day, duration, campaign, pdays, previous,
                job, marital, education, default_, housing, loan,
                contact, month, poutcome):
    row = {c: 0.0 for c in FEATURE_COLS}
    row['num__age']      = age
    row['num__balance']  = balance
    row['num__day']      = day
    row['num__duration'] = duration
    row['num__campaign'] = campaign
    row['num__pdays']    = pdays
    row['num__previous'] = previous
    row[f'cat__job_{job}']            = 1.0
    row[f'cat__marital_{marital}']    = 1.0
    row[f'cat__education_{education}']= 1.0
    row[f'cat__default_{default_}']   = 1.0
    row[f'cat__housing_{housing}']    = 1.0
    row[f'cat__loan_{loan}']          = 1.0
    row[f'cat__contact_{contact}']    = 1.0
    row[f'cat__month_{month}']        = 1.0
    row[f'cat__poutcome_{poutcome}']  = 1.0
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar Input ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Bank Deposit Prediction")
    st.markdown("---")

    st.markdown("### 👤 Customer Information")
    with st.expander("▼ Customer Profile", expanded=True):
        age       = st.number_input("Age", 18, 95, 40)
        job       = st.selectbox("Job", ['admin.','blue-collar','entrepreneur',
                                          'housemaid','management','retired',
                                          'self-employed','services','student',
                                          'technician','unemployed','unknown'])
        marital   = st.selectbox("Marital Status", ['married','single','divorced'])
        education = st.selectbox("Education", ['primary','secondary','tertiary','unknown'])
        default_  = st.selectbox("Credit Default", ['no','yes'])

    st.markdown("### 💰 Financial Information")
    with st.expander("▼ Financial Profile", expanded=True):
        balance  = st.number_input("Account Balance (€)", -10000, 100000, 1000)
        housing  = st.selectbox("Housing Loan", ['no','yes'])
        loan     = st.selectbox("Personal Loan", ['no','yes'])

    st.markdown("### 📞 Campaign Information")
    with st.expander("▼ Campaign Details", expanded=True):
        contact  = st.selectbox("Contact Type", ['cellular','telephone','unknown'])
        day      = st.slider("Last Contact Day", 1, 31, 15)
        month    = st.selectbox("Month", ['jan','feb','mar','apr','may','jun',
                                           'jul','aug','sep','oct','nov','dec'])
        duration = st.number_input("Call Duration (seconds)", 0, 5000, 300)
        campaign = st.number_input("Number of Contacts", 1, 50, 1)
        pdays    = st.number_input("Days Since Previous Contact (-1 = never)", -1, 999, -1)
        previous = st.number_input("Previous Contacts", 0, 50, 0)
        poutcome = st.selectbox("Previous Campaign Outcome",
                                 ['unknown','failure','other','success'])

    predict_btn = st.button("🔍 Predict", use_container_width=True)

# ── Hitung prediksi ───────────────────────────────────────────
if predict_btn:
    X_input = build_input(age, balance, day, duration, campaign, pdays, previous,
                          job, marital, education, default_, housing, loan,
                          contact, month, poutcome)
    proba = model.predict_proba(X_input)[0][1]
    score = int(proba * 100)
    if proba >= 0.7:
        category = "Very Likely"
    elif proba >= 0.4:
        category = "Likely"
    elif proba >= 0.2:
        category = "Unlikely"
    else:
        category = "Very Unlikely"

    st.session_state.history.append({
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Age': age, 'Job': job, 'Probability': f"{proba:.1%}",
        'Score': score, 'Category': category
    })
    st.session_state.last_pred = {
        'proba': proba, 'score': score, 'category': category,
        'age': age, 'job': job, 'marital': marital,
        'education': education, 'default': default_,
        'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month,
        'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview", "🔍 Prediction", "📊 Analytics",
    "🧠 Explainability", "📜 History"
])

# ════════════════════════════════════════════════════════
# TAB 1: OVERVIEW — Biru
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style='background: linear-gradient(90deg,#1a73e8,#4fc3f7);
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>🏦 Bank Deposit Prediction Dashboard</h1>
    <p style='color:#e3f2fd; margin:6px 0 0;'>
    Predict whether a customer will subscribe to a term deposit using XGBoost.</p>
    </div>""", unsafe_allow_html=True)

    subscribe_yes = int((df['deposit'] == 'yes').sum()) if 'deposit' in df.columns else 5289
    subscribe_no  = int((df['deposit'] == 'no').sum())  if 'deposit' in df.columns else 5873

    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in zip(
        [c1, c2, c3, c4],
        ["👥 Customers", "🔢 Features", "✅ Subscribe Yes", "❌ Subscribe No"],
        [f"{len(df):,}", "16", f"{subscribe_yes:,}", f"{subscribe_no:,}"],
        ["#1a73e8", "#0288d1", "#34a853", "#e53935"]
    ):
        col.markdown(f"""
        <div style='background:white; border-left:6px solid {color};
        border-radius:10px; padding:16px; text-align:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <p style='color:{color}; font-size:13px; margin:0; font-weight:600;'>{label}</p>
        <h2 style='color:#1a1a2e; margin:4px 0 0; font-size:28px;'>{val}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Pipeline Overview")
    st.markdown("""
    | Step | Numerik | Kategorikal |
    |------|---------|-------------|
    | 1 | SimpleImputer (median) | SimpleImputer (most_frequent) |
    | 2 | StandardScaler | OneHotEncoder |
    | 3 | → XGBClassifier | → XGBClassifier |
    """)
    st.markdown("### 📄 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2: PREDICTION — Hijau
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='background: linear-gradient(90deg,#34a853,#81c995);
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>🔍 Prediction</h1>
    <p style='color:#e8f5e9; margin:6px 0 0;'>
    Input data nasabah di sidebar, lalu klik <b>Predict</b>.</p>
    </div>""", unsafe_allow_html=True)

    if 'last_pred' in st.session_state:
        p = st.session_state.last_pred
        proba_val = p['proba']
        score_val = p['score']
        cat_val   = p['category']

        color_map = {
            "Very Likely": "#34a853", "Likely": "#1a73e8",
            "Unlikely": "#fa7b17",    "Very Unlikely": "#e53935"
        }
        card_color = color_map.get(cat_val, "#1a73e8")

        col1, col2, col3 = st.columns(3)
        for col, label, val in zip(
            [col1, col2, col3],
            ["📊 Probability", "🏆 Score", "🏷️ Category"],
            [f"{proba_val:.1%}", f"{score_val}/100", cat_val]
        ):
            col.markdown(f"""
            <div style='background:white; border-left:6px solid {card_color};
            border-radius:10px; padding:16px; text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
            <p style='color:{card_color}; font-size:13px; margin:0; font-weight:600;'>{label}</p>
            <h2 style='color:#1a1a2e; margin:4px 0 0; font-size:26px;'>{val}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(proba_val)

        st.markdown("---")
        st.markdown("### 💡 Recommendation")
        rec_map = {
            "Very Likely": ("✅", "#34a853", "High potential customer. Prioritize immediate follow-up."),
            "Likely":      ("📞", "#1a73e8", "Potential customer. Additional follow-up suggested."),
            "Unlikely":    ("⚠️", "#fa7b17", "Low potential. Consider alternative products."),
            "Very Unlikely":("❌","#e53935", "Very low potential. Deprioritize for deposit campaign.")
        }
        icon, rcolor, msg = rec_map[cat_val]
        st.markdown(f"""
        <div style='background:{rcolor}18; border-left:5px solid {rcolor};
        border-radius:8px; padding:14px 18px;'>
        <b style='color:{rcolor};'>{icon} {msg}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 👤 Customer Input Summary")
        summary = pd.DataFrame([{
            'age': p['age'], 'job': p['job'], 'marital': p['marital'],
            'education': p['education'], 'default': p['default'],
            'balance': p['balance'], 'housing': p['housing'], 'loan': p['loan'],
            'contact': p['contact'], 'day': p['day'], 'month': p['month'],
            'duration': p['duration'], 'campaign': p['campaign'],
            'pdays': p['pdays'], 'previous': p['previous'], 'poutcome': p['poutcome']
        }])
        st.dataframe(summary, use_container_width=True)
    else:
        st.markdown("""
        <div style='background:#e8f5e9; border-left:5px solid #34a853;
        border-radius:8px; padding:14px 18px;'>
        👈 <b>Silakan isi data nasabah di sidebar dan klik Predict.</b>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3: ANALYTICS — Orange
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='background: linear-gradient(90deg,#fa7b17,#ffb74d);
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>📊 Analytics Dashboard</h1>
    <p style='color:#fff3e0; margin:6px 0 0;'>
    Distribusi dan analisis demografi nasabah.</p>
    </div>""", unsafe_allow_html=True)

    PALETTE = ['#1a73e8','#fa7b17','#34a853','#e53935','#9c27b0','#00bcd4']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Job Distribution**")
        if 'job' in df.columns and 'loan' in df.columns:
            job_counts = df.groupby(['job','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            job_counts.plot(kind='bar', ax=ax, color=['#1a73e8','#fa7b17'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=45)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Job vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Marital Distribution**")
        if 'marital' in df.columns and 'loan' in df.columns:
            mar_counts = df.groupby(['marital','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            mar_counts.plot(kind='bar', ax=ax, color=['#34a853','#e53935'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Marital vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col3:
        st.markdown("**Contact Type**")
        if 'contact' in df.columns:
            contact_counts = df['contact'].value_counts()
            fig, ax = plt.subplots(figsize=(5,4))
            ax.pie(contact_counts, labels=contact_counts.index,
                   autopct='%1.1f%%', colors=['#1a73e8','#fa7b17','#34a853'],
                   startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Contact Type Distribution"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("**Balance Distribution**")
        if 'balance' in df.columns:
            fig, ax = plt.subplots(figsize=(5,4))
            df['balance'].clip(-2000, 10000).hist(bins=40, ax=ax,
                color='#9c27b0', edgecolor='white', alpha=0.85)
            ax.set_xlabel("Balance (€)"); ax.set_ylabel("Count")
            ax.set_title("Account Balance Distribution"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col5:
        st.markdown("**Contact Month Trend**")
        if 'month' in df.columns:
            month_order = ['jan','feb','mar','apr','may','jun',
                           'jul','aug','sep','oct','nov','dec']
            month_counts = df['month'].value_counts().reindex(month_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            ax.fill_between(range(len(month_counts)), month_counts.values,
                            alpha=0.3, color='#fa7b17')
            ax.plot(range(len(month_counts)), month_counts.values,
                    marker='o', color='#fa7b17', linewidth=2)
            ax.set_xticks(range(len(month_counts)))
            ax.set_xticklabels(month_order, rotation=45, fontsize=8)
            ax.set_ylabel("Count"); ax.set_title("Contact per Month")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col6:
        st.markdown("**Education Distribution**")
        if 'education' in df.columns and 'loan' in df.columns:
            edu_counts = df.groupby(['education','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            edu_counts.plot(kind='bar', ax=ax, color=['#00bcd4','#e53935'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Education vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**Previous Campaign Outcome**")
        if 'poutcome' in df.columns and 'loan' in df.columns:
            po_counts = df.groupby(['poutcome','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            po_counts.plot(kind='bar', ax=ax, color=['#9c27b0','#fa7b17'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Poutcome vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col8:
        st.markdown("**Deposit Distribution**")
        if 'deposit' in df.columns and 'loan' in df.columns:
            dep_counts = df.groupby(['deposit','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            dep_counts.plot(kind='bar', ax=ax, color=['#34a853','#e53935'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Deposit vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY — Ungu
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='background: linear-gradient(90deg,#9c27b0,#ce93d8);
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>🧠 Explainability Dashboard</h1>
    <p style='color:#f3e5f5; margin:6px 0 0;'>
    Feature importance dari model XGBoost.</p>
    </div>""", unsafe_allow_html=True)

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS, 'Importance': importance
    }).sort_values('Importance', ascending=False).head(15)

    purple_palette = plt.cm.Purples(np.linspace(0.4, 0.9, len(feat_df)))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
                   color=purple_palette)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 15 Most Important Features (XGBoost)", fontsize=13, fontweight='bold')
    for bar, val in zip(bars, feat_df['Importance'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 📋 Full Feature Importance Table")
    full_feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS, 'Importance': importance
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    full_feat_df.index += 1
    st.dataframe(full_feat_df, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 5: HISTORY — Merah
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style='background: linear-gradient(90deg,#e53935,#ef9a9a);
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>📜 Prediction History</h1>
    <p style='color:#ffebee; margin:6px 0 0;'>
    Riwayat semua prediksi dalam sesi ini.</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        avg_score = hist_df['Score'].mean()

        c1, c2 = st.columns(2)
        for col, label, val, color in zip(
            [c1, c2],
            ["📊 Total Predictions", "🏆 Average Score"],
            [str(len(hist_df)), f"{avg_score:.1f}"],
            ["#e53935", "#fa7b17"]
        ):
            col.markdown(f"""
            <div style='background:white; border-left:6px solid {color};
            border-radius:10px; padding:16px; text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
            <p style='color:{color}; font-size:13px; margin:0; font-weight:600;'>{label}</p>
            <h2 style='color:#1a1a2e; margin:4px 0 0; font-size:28px;'>{val}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            if 'last_pred' in st.session_state:
                del st.session_state.last_pred
            st.rerun()
    else:
        st.markdown("""
        <div style='background:#ffebee; border-left:5px solid #e53935;
        border-radius:8px; padding:14px 18px;'>
        📭 <b>Belum ada riwayat prediksi. Lakukan prediksi terlebih dahulu.</b>
        </div>""", unsafe_allow_html=True)
