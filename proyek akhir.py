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

# ── Tema Biru & Pink ──────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a4e 0%, #2d1b69 50%, #8b1a6b 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #f8bbd0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #e91e8c, #1565c0);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 16px;
    padding: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #fce4ec;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 14px;
    color: #880e4f;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #1565c0, #e91e8c) !important;
    color: white !important;
}
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

# ── Kolom fitur ───────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────
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
        balance = st.number_input("Account Balance (€)", -10000, 100000, 1000)
        housing = st.selectbox("Housing Loan", ['no','yes'])
        loan    = st.selectbox("Personal Loan", ['no','yes'])

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

# ── Prediksi ──────────────────────────────────────────────────
if predict_btn:
    X_input = build_input(age, balance, day, duration, campaign, pdays, previous,
                          job, marital, education, default_, housing, loan,
                          contact, month, poutcome)
    proba = model.predict_proba(X_input)[0][1]
    score = int(proba * 100)
    category = ("Very Likely" if proba >= 0.7 else
                "Likely"      if proba >= 0.4 else
                "Unlikely"    if proba >= 0.2 else "Very Unlikely")
    st.session_state.history.append({
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Age': age, 'Job': job,
        'Probability': f"{proba:.1%}", 'Score': score, 'Category': category
    })
    st.session_state.last_pred = dict(
        proba=proba, score=score, category=category,
        age=age, job=job, marital=marital, education=education,
        default=default_, balance=balance, housing=housing, loan=loan,
        contact=contact, day=day, month=month, duration=duration,
        campaign=campaign, pdays=pdays, previous=previous, poutcome=poutcome
    )

# ── Helper: metric card ───────────────────────────────────────
def metric_card(label, val, color):
    st.markdown(f"""
    <div style='background:white; border-left:6px solid {color};
    border-radius:10px; padding:16px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:8px;'>
    <p style='color:{color}; font-size:13px; margin:0; font-weight:600;'>{label}</p>
    <h2 style='color:#1a1a2e; margin:4px 0 0; font-size:26px;'>{val}</h2>
    </div>""", unsafe_allow_html=True)

def header_banner(title, subtitle, grad_left, grad_right):
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,{grad_left},{grad_right});
    padding:20px 28px; border-radius:14px; margin-bottom:20px;'>
    <h1 style='color:white; margin:0;'>{title}</h1>
    <p style='color:rgba(255,255,255,0.85); margin:6px 0 0;'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

# Palet warna biru-pink untuk chart
BLUE  = '#1565c0'
PINK  = '#e91e8c'
LBLUE = '#42a5f5'
LPINK = '#f48fb1'
DBLUE = '#0d47a1'
DPINK = '#880e4f'

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview", "🔍 Prediction", "📊 Analytics",
    "🧠 Explainability", "📜 History"
])

# ════ TAB 1: OVERVIEW ════════════════════════════════════════
with tab1:
    header_banner("🏦 Bank Deposit Prediction Dashboard",
                  "Predict whether a customer will subscribe to a term deposit using XGBoost.",
                  BLUE, PINK)

    subscribe_yes = int((df['deposit']=='yes').sum()) if 'deposit' in df.columns else 5289
    subscribe_no  = int((df['deposit']=='no').sum())  if 'deposit' in df.columns else 5873

    st.markdown("### 📋 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("👥 Customers",   f"{len(df):,}", BLUE)
    with c2: metric_card("🔢 Features",    "16",           PINK)
    with c3: metric_card("✅ Subscribe Yes",f"{subscribe_yes:,}", LBLUE)
    with c4: metric_card("❌ Subscribe No", f"{subscribe_no:,}",  LPINK)

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

# ════ TAB 2: PREDICTION ══════════════════════════════════════
with tab2:
    header_banner("🔍 Prediction",
                  "Input data nasabah di sidebar, lalu klik <b>Predict</b>.",
                  PINK, BLUE)

    if 'last_pred' in st.session_state:
        p = st.session_state.last_pred
        cat_color = {
            "Very Likely": BLUE, "Likely": LBLUE,
            "Unlikely": LPINK,   "Very Unlikely": PINK
        }.get(p['category'], BLUE)

        c1,c2,c3 = st.columns(3)
        with c1: metric_card("📊 Probability", f"{p['proba']:.1%}", PINK)
        with c2: metric_card("🏆 Score",        f"{p['score']}/100", BLUE)
        with c3: metric_card("🏷️ Category",     p['category'],       cat_color)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,{BLUE}22,{PINK}22);
        border-radius:10px; padding:4px 10px;'>
        </div>""", unsafe_allow_html=True)
        st.progress(p['proba'])

        st.markdown("---")
        st.markdown("### 💡 Recommendation")
        rec = {
            "Very Likely":  (BLUE, "✅ High potential customer. Prioritize immediate follow-up."),
            "Likely":       (LBLUE,"📞 Potential customer. Additional follow-up suggested."),
            "Unlikely":     (LPINK,"⚠️ Low potential. Consider alternative products."),
            "Very Unlikely":(PINK, "❌ Very low potential. Deprioritize for deposit campaign.")
        }[p['category']]
        st.markdown(f"""
        <div style='background:{rec[0]}18; border-left:5px solid {rec[0]};
        border-radius:8px; padding:14px 18px;'>
        <b style='color:{rec[0]};'>{rec[1]}</b></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 👤 Customer Input Summary")
        st.dataframe(pd.DataFrame([{
            'age':p['age'],'job':p['job'],'marital':p['marital'],
            'education':p['education'],'default':p['default'],
            'balance':p['balance'],'housing':p['housing'],'loan':p['loan'],
            'contact':p['contact'],'day':p['day'],'month':p['month'],
            'duration':p['duration'],'campaign':p['campaign'],
            'pdays':p['pdays'],'previous':p['previous'],'poutcome':p['poutcome']
        }]), use_container_width=True)
    else:
        st.markdown(f"""
        <div style='background:{BLUE}11; border-left:5px solid {BLUE};
        border-radius:8px; padding:14px 18px;'>
        👈 <b>Silakan isi data nasabah di sidebar dan klik Predict.</b>
        </div>""", unsafe_allow_html=True)

# ════ TAB 3: ANALYTICS ═══════════════════════════════════════
with tab3:
    header_banner("📊 Analytics Dashboard",
                  "Distribusi dan analisis demografi nasabah.",
                  DPINK, LBLUE)

    def bp_chart(ax, data, title, colors=[BLUE, PINK]):
        data.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel(""); ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(['No Loan','Loan'], fontsize=8)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Job Distribution**")
        if 'job' in df.columns and 'loan' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            bp_chart(ax, df.groupby(['job','loan']).size().unstack(fill_value=0),
                     "Job vs Loan")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown("**Marital Distribution**")
        if 'marital' in df.columns and 'loan' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            bp_chart(ax, df.groupby(['marital','loan']).size().unstack(fill_value=0),
                     "Marital vs Loan", [LBLUE, LPINK])
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        st.markdown("**Contact Type**")
        if 'contact' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            cc = df['contact'].value_counts()
            ax.pie(cc, labels=cc.index, autopct='%1.1f%%',
                   colors=[BLUE, PINK, LBLUE], startangle=90,
                   wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Contact Type", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    c4,c5,c6 = st.columns(3)
    with c4:
        st.markdown("**Balance Distribution**")
        if 'balance' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            n, bins, patches = ax.hist(df['balance'].clip(-2000,10000),
                                       bins=40, edgecolor='white')
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.cool(i / len(patches)))
            ax.set_xlabel("Balance (€)"); ax.set_title("Balance Distribution", fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c5:
        st.markdown("**Contact Month Trend**")
        if 'month' in df.columns:
            month_order = ['jan','feb','mar','apr','may','jun',
                           'jul','aug','sep','oct','nov','dec']
            mc = df['month'].value_counts().reindex(month_order, fill_value=0)
            fig,ax = plt.subplots(figsize=(5,4))
            ax.fill_between(range(len(mc)), mc.values, alpha=0.25, color=PINK)
            ax.plot(range(len(mc)), mc.values, marker='o', color=BLUE,
                    linewidth=2, markerfacecolor=PINK, markeredgecolor=BLUE, markersize=7)
            ax.set_xticks(range(len(mc))); ax.set_xticklabels(month_order, rotation=45, fontsize=8)
            ax.set_title("Contact per Month", fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c6:
        st.markdown("**Education Distribution**")
        if 'education' in df.columns and 'loan' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            bp_chart(ax, df.groupby(['education','loan']).size().unstack(fill_value=0),
                     "Education vs Loan", [DBLUE, LPINK])
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    c7,c8 = st.columns(2)
    with c7:
        st.markdown("**Previous Campaign Outcome**")
        if 'poutcome' in df.columns and 'loan' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            bp_chart(ax, df.groupby(['poutcome','loan']).size().unstack(fill_value=0),
                     "Poutcome vs Loan", [BLUE, PINK])
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c8:
        st.markdown("**Deposit Distribution**")
        if 'deposit' in df.columns and 'loan' in df.columns:
            fig,ax = plt.subplots(figsize=(5,4))
            bp_chart(ax, df.groupby(['deposit','loan']).size().unstack(fill_value=0),
                     "Deposit vs Loan", [LBLUE, LPINK])
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ════ TAB 4: EXPLAINABILITY ══════════════════════════════════
with tab4:
    header_banner("🧠 Explainability Dashboard",
                  "Feature importance dari model XGBoost.",
                  DBLUE, DPINK)

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS, 'Importance': importance
    }).sort_values('Importance', ascending=False).head(15)

    # Gradient biru → pink untuk bar chart
    n = len(feat_df)
    bar_colors = [
        (
            int(21  + (233-21)  * i/(n-1)),  # R: biru(21) → pink(233)
            int(101 + (30-101)  * i/(n-1)),  # G: biru(101)→ pink(30)
            int(192 + (140-192) * i/(n-1))   # B: biru(192)→ pink(140)
        )
        for i in range(n)
    ]
    bar_colors_hex = [f'#{r:02x}{g:02x}{b:02x}' for r,g,b in bar_colors]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
                   color=bar_colors_hex[::-1])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 15 Most Important Features (XGBoost)",
                 fontsize=13, fontweight='bold')
    for bar, val in zip(bars, feat_df['Importance'][::-1]):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### 📋 Full Feature Importance Table")
    full_feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS, 'Importance': importance
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    full_feat_df.index += 1
    st.dataframe(full_feat_df, use_container_width=True)

# ════ TAB 5: HISTORY ═════════════════════════════════════════
with tab5:
    header_banner("📜 Prediction History",
                  "Riwayat semua prediksi dalam sesi ini.",
                  PINK, DBLUE)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        c1, c2 = st.columns(2)
        with c1: metric_card("📊 Total Predictions", str(len(hist_df)), BLUE)
        with c2: metric_card("🏆 Average Score", f"{hist_df['Score'].mean():.1f}", PINK)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            if 'last_pred' in st.session_state:
                del st.session_state.last_pred
            st.rerun()
    else:
        st.markdown(f"""
        <div style='background:{PINK}11; border-left:5px solid {PINK};
        border-radius:8px; padding:14px 18px;'>
        📭 <b>Belum ada riwayat prediksi. Lakukan prediksi terlebih dahulu.</b>
        </div>""", unsafe_allow_html=True)
