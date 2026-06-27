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

# ── Kolom fitur (51 hasil OHE, urutan training) ──────────────
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

# ── Session state untuk history ───────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar Input ─────────────────────────────────────────────
with st.sidebar:
    st.header("🏦 Bank Deposit Prediction")
    st.markdown("---")

    st.subheader("👤 Customer Information")
    with st.expander("▼ Customer Profile", expanded=True):
        age       = st.number_input("Age", 18, 95, 40)
        job       = st.selectbox("Job", ['admin.','blue-collar','entrepreneur',
                                          'housemaid','management','retired',
                                          'self-employed','services','student',
                                          'technician','unemployed','unknown'])
        marital   = st.selectbox("Marital Status", ['married','single','divorced'])
        education = st.selectbox("Education", ['primary','secondary','tertiary','unknown'])
        default_  = st.selectbox("Credit Default", ['no','yes'])

    st.subheader("💰 Financial Information")
    with st.expander("▼ Financial Profile", expanded=True):
        balance  = st.number_input("Account Balance (€)", -10000, 100000, 1000)
        housing  = st.selectbox("Housing Loan", ['no','yes'])
        loan     = st.selectbox("Personal Loan", ['no','yes'])

    st.subheader("📞 Campaign Information")
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
proba = None
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
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:
    st.title("🏦 Bank Deposit Prediction Dashboard")
    st.markdown("Predict whether a customer will subscribe to a term deposit using XGBoost.")
    st.markdown("---")

    st.subheader("Dataset Overview")
    subscribe_yes = int((df['deposit'] == 'yes').sum()) if 'deposit' in df.columns else 5289
    subscribe_no  = int((df['deposit'] == 'no').sum())  if 'deposit' in df.columns else 5873

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Features", "16")
    c3.metric("Subscribe Yes", f"{subscribe_yes:,}")
    c4.metric("Subscribe No",  f"{subscribe_no:,}")

    st.markdown("---")
    st.subheader("📋 Pipeline Overview")
    st.markdown("""
    | Step | Numerik | Kategorikal |
    |------|---------|-------------|
    | 1 | SimpleImputer (median) | SimpleImputer (most_frequent) |
    | 2 | StandardScaler | OneHotEncoder |
    | 3 | → XGBClassifier | → XGBClassifier |
    """)

    st.markdown("---")
    st.subheader("📄 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2: PREDICTION
# ════════════════════════════════════════════════════════
with tab2:
    st.title("🔍 Prediction")
    st.markdown("Input data nasabah di sidebar, lalu klik **Predict**.")

    if 'last_pred' in st.session_state:
        p = st.session_state.last_pred
        proba_val = p['proba']
        score_val = p['score']
        cat_val   = p['category']

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability", f"{proba_val:.1%}")
        col2.metric("Score", f"{score_val}/100")
        if cat_val in ["Very Likely", "Likely"]:
            col3.success(f"🟢 {cat_val}")
        else:
            col3.error(f"🔴 {cat_val}")

        # Progress bar
        st.progress(proba_val)

        st.markdown("---")
        st.subheader("💡 Recommendation")
        if cat_val == "Very Likely":
            st.success("✅ High potential customer. Prioritize immediate follow-up.")
        elif cat_val == "Likely":
            st.info("📞 Potential customer. Additional follow-up suggested.")
        elif cat_val == "Unlikely":
            st.warning("⚠️ Low potential. Consider alternative products.")
        else:
            st.error("❌ Very low potential. Deprioritize for deposit campaign.")

        st.markdown("---")
        st.subheader("👤 Customer Input Summary")
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
        st.info("👈 Silakan isi data nasabah di sidebar dan klik **Predict**.")

# ════════════════════════════════════════════════════════
# TAB 3: ANALYTICS
# ════════════════════════════════════════════════════════
with tab3:
    st.title("📊 Analytics Dashboard")
    st.markdown("Distribusi dan analisis demografi nasabah.")

    # Baris 1
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Job Distribution")
        if 'job' in df.columns and 'loan' in df.columns:
            job_counts = df.groupby(['job','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            job_counts.plot(kind='bar', ax=ax, color=['#2196F3','#FF9800'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=45)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Job vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Marital Distribution")
        if 'marital' in df.columns and 'loan' in df.columns:
            mar_counts = df.groupby(['marital','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            mar_counts.plot(kind='bar', ax=ax, color=['#2196F3','#FF9800'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Marital vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col3:
        st.subheader("Contact Type")
        if 'contact' in df.columns:
            contact_counts = df['contact'].value_counts()
            fig, ax = plt.subplots(figsize=(5,4))
            ax.pie(contact_counts, labels=contact_counts.index,
                   autopct='%1.1f%%', colors=['#2196F3','#FF9800','#4CAF50'])
            ax.set_title("Contact Type Distribution"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Baris 2
    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Balance Distribution")
        if 'balance' in df.columns:
            fig, ax = plt.subplots(figsize=(5,4))
            df['balance'].clip(-2000, 10000).hist(bins=40, ax=ax, color='#2196F3', edgecolor='white')
            ax.set_xlabel("Balance (€)"); ax.set_ylabel("Count")
            ax.set_title("Account Balance Distribution"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col5:
        st.subheader("Contact Month Trend")
        if 'month' in df.columns:
            month_order = ['jan','feb','mar','apr','may','jun',
                           'jul','aug','sep','oct','nov','dec']
            month_counts = df['month'].value_counts().reindex(month_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            month_counts.plot(kind='line', marker='o', ax=ax, color='#2196F3')
            ax.set_xlabel("Month"); ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            ax.set_title("Contact per Month"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col6:
        st.subheader("Education Distribution")
        if 'education' in df.columns and 'loan' in df.columns:
            edu_counts = df.groupby(['education','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            edu_counts.plot(kind='bar', ax=ax, color=['#2196F3','#FF9800'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Education vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Baris 3
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("Previous Campaign Outcome")
        if 'poutcome' in df.columns and 'loan' in df.columns:
            po_counts = df.groupby(['poutcome','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            po_counts.plot(kind='bar', ax=ax, color=['#2196F3','#FF9800'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Poutcome vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col8:
        st.subheader("Deposit Distribution")
        if 'deposit' in df.columns and 'loan' in df.columns:
            dep_counts = df.groupby(['deposit','loan']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,4))
            dep_counts.plot(kind='bar', ax=ax, color=['#2196F3','#FF9800'])
            ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            ax.legend(['No Loan','Loan'], fontsize=8)
            ax.set_title("Deposit vs Loan"); plt.tight_layout()
            st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY
# ════════════════════════════════════════════════════════
with tab4:
    st.title("🧠 Explainability Dashboard")
    st.markdown("Feature importance dari model XGBoost.")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
                   color='#2196F3')
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 15 Most Important Features (XGBoost)")
    for bar, val in zip(bars, feat_df['Importance'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Full Feature Importance Table")
    full_feat_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    full_feat_df.index += 1
    st.dataframe(full_feat_df, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 5: HISTORY
# ════════════════════════════════════════════════════════
with tab5:
    st.title("📜 Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        total   = len(hist_df)
        avg_score = hist_df['Score'].mean()

        c1, c2 = st.columns(2)
        c1.metric("Total Predictions", total)
        c2.metric("Average Score", f"{avg_score:.1f}")

        st.markdown("---")
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            if 'last_pred' in st.session_state:
                del st.session_state.last_pred
            st.rerun()
    else:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi terlebih dahulu.")
