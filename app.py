import os
import sys
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from predict import predict_text, MODELS
from data_preprocessing import clean_text

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Data loaders ----------
@st.cache_data
def load_processed():
    path = os.path.join(BASE_DIR, "data", "processed", "sms_spam_cleaned.csv")
    return pd.read_csv(path)

@st.cache_data
def load_comparison():
    path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
    return pd.read_csv(path)

@st.cache_data
def load_test_split():
    return joblib.load(os.path.join(BASE_DIR, "data", "processed", "test_split.pkl"))

@st.cache_data
def compute_word_freq():
    df = load_processed()
    spam_words = " ".join(df[df.label_num == 1]["cleaned_text"].fillna("").astype(str)).split()
    ham_words = " ".join(df[df.label_num == 0]["cleaned_text"].fillna("").astype(str)).split()
    return Counter(spam_words), Counter(ham_words)

@st.cache_data
def compute_confusion_matrices():
    X_test, y_test, X_test_vec = load_test_split()
    out = {}
    for name, fname in [("Naive Bayes", "naive_bayes.pkl"),
                        ("Logistic Regression", "logistic_regression.pkl"),
                        ("Linear SVM", "svm.pkl")]:
        model = joblib.load(os.path.join(BASE_DIR, "models", fname))
        preds = model.predict(X_test_vec)
        from sklearn.metrics import confusion_matrix
        out[name] = confusion_matrix(y_test, preds)
    return out, y_test

# ---------- CSS ----------
st.markdown("""
<style>
:root {
  --bg-primary: #080B18;
  --bg-secondary: #10162A;
  --card-bg: #151D35;
  --accent: #7C5CFC;
  --accent2: #00D4FF;
  --success: #16D9A6;
  --danger: #FF5277;
  --warning: #FFB84D;
  --text: #F4F6FF;
  --text2: #A6AEC8;
}
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: var(--text);
}
.stApp {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(124,92,252,0.18), transparent 60%),
              radial-gradient(1000px 500px at 90% 10%, rgba(0,212,255,0.14), transparent 60%),
              linear-gradient(180deg, #080B18 0%, #0B1022 100%);
  background-attachment: fixed;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1226 0%, #080B18 100%);
  border-right: 1px solid rgba(124,92,252,0.15);
}
/* animations */
@keyframes fadeInUp { from { opacity:0; transform: translateY(18px);} to { opacity:1; transform: translateY(0);} }
@keyframes fadeIn { from { opacity:0;} to { opacity:1;} }
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes floaty {
  0%,100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}
@keyframes pulseGlow {
  0%,100% { box-shadow: 0 0 0 0 rgba(124,92,252,0.4); }
  50% { box-shadow: 0 0 24px 4px rgba(124,92,252,0.25); }
}
.fade-in { animation: fadeIn 0.6s ease-out both; }
.slide-up { animation: fadeInUp 0.7s ease-out both; }
.float { animation: floaty 4s ease-in-out infinite; }

/* hero */
.hero {
  position: relative;
  padding: 2.6rem 2rem;
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(124,92,252,0.25), rgba(0,212,255,0.18));
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
  margin-bottom: 1.5rem;
}
.hero::before {
  content: "";
  position: absolute; inset: -40%;
  background: conic-gradient(from 180deg, rgba(124,92,252,0.15), rgba(0,212,255,0.1), rgba(124,92,252,0.05), transparent);
  animation: gradientShift 12s ease infinite;
  background-size: 200% 200%;
  filter: blur(40px);
}
.hero-content { position: relative; z-index: 2; }
.badge {
  display: inline-block;
  padding: 6px 14px;
  border-radius: 999px;
  background: rgba(124,92,252,0.2);
  border: 1px solid rgba(124,92,252,0.5);
  color: #D9CEFF;
  font-size: 0.78rem;
  letter-spacing: 1.5px;
  font-weight: 600;
  margin-bottom: 0.8rem;
}
.hero h1 {
  font-size: 2.6rem;
  font-weight: 800;
  margin: 0.2rem 0 0.4rem 0;
  background: linear-gradient(90deg, #F4F6FF, #7C5CFC, #00D4FF);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero p { color: var(--text2); font-size: 1.05rem; max-width: 720px; }
.hero-icon { font-size: 3.2rem; animation: floaty 3.5s ease-in-out infinite; display:inline-block; }

/* metric cards */
.metric-card {
  background: rgba(21,29,53,0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 1.3rem 1.2rem;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  animation: fadeInUp 0.7s ease-out both;
  position: relative;
  overflow: hidden;
}
.metric-card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(0,0,0,0.35); }
.metric-card .icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.metric-card .label { color: var(--text2); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 2rem; font-weight: 800; margin: 0.2rem 0; }
.metric-card .sub { color: var(--text2); font-size: 0.8rem; }

/* glass card */
.glass {
  background: rgba(21,29,53,0.6);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 1.4rem;
  transition: transform 0.25s ease;
  animation: fadeInUp 0.7s ease-out both;
}
.glass:hover { transform: translateY(-3px); }

/* buttons */
.stButton > button {
  background: linear-gradient(135deg, #7C5CFC, #00D4FF) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.6rem 1.4rem !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 8px 24px rgba(124,92,252,0.35) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 32px rgba(0,212,255,0.45) !important;
}
.stTextArea textarea {
  background: rgba(21,29,53,0.8) !important;
  border: 1px solid rgba(124,92,252,0.3) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
}
.stTextArea textarea:focus { border-color: #00D4FF !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.2) !important; }

/* result cards */
.result-spam {
  background: linear-gradient(135deg, rgba(255,82,119,0.18), rgba(255,184,77,0.12));
  border: 1px solid rgba(255,82,119,0.4);
  border-radius: 20px;
  padding: 1.8rem;
  animation: fadeInUp 0.5s ease-out both;
}
.result-ham {
  background: linear-gradient(135deg, rgba(22,217,166,0.18), rgba(0,212,255,0.12));
  border: 1px solid rgba(22,217,166,0.4);
  border-radius: 20px;
  padding: 1.8rem;
  animation: fadeInUp 0.5s ease-out both;
}
.result-title { font-size: 1.9rem; font-weight: 800; margin: 0.4rem 0; }
.result-spam .result-title { color: #FF7A99; }
.result-ham .result-title { color: #3DE6B8; }

/* sidebar nav */
.sidebar-title { font-size: 1.1rem; font-weight: 700; color: #D9CEFF; }
.sidebar-sub { color: var(--text2); font-size: 0.85rem; }

/* responsive */
@media (max-width: 768px) {
  .hero h1 { font-size: 1.8rem; }
  .metric-card .value { font-size: 1.5rem; }
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.6rem 0 1rem 0;">
      <div style="font-size:2.4rem;" class="float">🛡️</div>
      <div class="sidebar-title">SpamGuard AI</div>
      <div class="sidebar-sub">NLP Security Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Dashboard", "🔍 Spam Detection", "📊 Model Comparison", "📈 Data Insights", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    model_name = st.selectbox("Active Model", list(MODELS.keys()), index=2)
    cmp_df = load_comparison()
    best_row = cmp_df.sort_values("F1-Score", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="glass" style="padding:0.9rem;">
      <div style="color:#A6AEC8; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Best Model</div>
      <div style="font-size:1.05rem; font-weight:700; color:#16D9A6; margin:0.2rem 0;">{best_row['Model']}</div>
      <div style="color:#A6AEC8; font-size:0.8rem;">F1: {best_row['F1-Score']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="color:#A6AEC8; font-size:0.78rem; line-height:1.5;">
      <b>Project</b><br>Email Spam Detection<br>Using NLP & ML<br><br>
      <b>Tech</b><br>Python · Streamlit · scikit-learn · NLTK
    </div>
    """, unsafe_allow_html=True)

# ---------- Helpers ----------
def svg_icon(name):
    icons = {
        "mail": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M2 7l10 7 10-7"/></svg>',
        "shield": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l8 4v6c0 5-3.5 9-8 10-4.5-1-8-5-8-10V6l8-4z"/></svg>',
        "chart": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>',
        "database": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
    }
    return icons.get(name, "")

df_proc = load_processed()
spam_count = int((df_proc.label_num == 1).sum())
ham_count = int((df_proc.label_num == 0).sum())
total_count = len(df_proc)
cmp_df = load_comparison()

# ---------- Page: Dashboard ----------
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero slide-up">
      <div class="hero-content">
        <div class="badge">✨ AI-POWERED SECURITY</div>
        <div class="hero-icon">📧</div>
        <h1>EMAIL SPAM DETECTION</h1>
        <p>Intelligent Message Classification Powered by NLP & Machine Learning.<br>
        Analyze messages in real time using TF-IDF features and trained ML models to detect spam and protect your inbox.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Messages", total_count, "#7C5CFC", "database", "analysed in dataset"),
        ("Spam Messages", spam_count, "#FF5277", "mail", "spam examples"),
        ("Legitimate Messages", ham_count, "#16D9A6", "shield", "ham examples"),
        ("Best Model", best_row['Model'], "#00D4FF", "chart", f"F1: {best_row['F1-Score']:.3f}"),
    ]
    for col, (title, val, color, icon, sub) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
              <div class="icon" style="color:{color};">{svg_icon(icon)}</div>
              <div class="label">{title}</div>
              <div class="value" style="color:{color};">{val}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown('<div class="glass slide-up"><h3 style="margin-top:0;">How it works</h3>'
                    '<p style="color:#A6AEC8;">Messages are lowercased, cleaned (HTML, URLs, emails removed), tokenized, '
                    'stopwords removed and lemmatized. TF-IDF converts text to numerical features, which three ML models '
                    '(Naive Bayes, Logistic Regression, Linear SVM) use to classify as Spam or Legitimate.</p></div>',
                    unsafe_allow_html=True)
    with colB:
        # mini distribution
        fig = px.pie(names=["Legitimate", "Spam"], values=[ham_count, spam_count],
                     color_discrete_sequence=["#16D9A6", "#FF5277"], hole=0.55)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#F4F6FF", margin=dict(t=10, b=10, l=10, r=10), height=260)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Page: Spam Detection ----------
elif page == "🔍 Spam Detection":
    st.markdown('<div class="fade-in"><h2>🔍 Spam Detection</h2>'
                '<p style="color:#A6AEC8;">Paste a message or email and classify it instantly.</p></div>',
                unsafe_allow_html=True)

    if "msg_input" not in st.session_state:
        st.session_state.msg_input = ""

    examples = [
        "Free entry in 2 a weekly comp to win FA Cup final tkts",
        "Congratulations! You won a free iPhone. Click here to claim now",
        "Hey, are we still meeting for lunch tomorrow?",
        "Your account has been suspended. Verify now to avoid closure",
    ]
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:28] + "…", key=f"ex_{i}"):
            st.session_state.msg_input = ex
            st.rerun()

    text = st.text_area("Message", height=160,
                        placeholder="Type or paste your message here...",
                        key="msg_input")

    if st.button("🚀 CLASSIFY MESSAGE"):
        if not text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing message..."):
                res = predict_text(text, model_name)
            label = res["label"]
            conf = res["confidence"] if res["confidence"] is not None else 0.0
            cleaned = res["cleaned"]
            if label == "Spam":
                st.markdown(f"""
                <div class="result-spam slide-up">
                  <div style="font-size:2.4rem;">⚠️</div>
                  <div class="result-title">SPAM DETECTED</div>
                  <p style="color:#FFB3C4;">This message shows spam indicators. Treat with caution.</p>
                  <div style="margin-top:0.8rem;">
                    <span style="color:#FF7A99; font-weight:700;">Model:</span> {model_name}<br>
                    <span style="color:#FF7A99; font-weight:700;">Score:</span> {conf:.2%} (model score, not calibrated probability for SVM)<br>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham slide-up">
                  <div style="font-size:2.4rem;">🛡️</div>
                  <div class="result-title">LEGITIMATE MESSAGE</div>
                  <p style="color:#A8F0DA;">This message appears legitimate.</p>
                  <div style="margin-top:0.8rem;">
                    <span style="color:#3DE6B8; font-weight:700;">Model:</span> {model_name}<br>
                    <span style="color:#3DE6B8; font-weight:700;">Score:</span> {conf:.2%}<br>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with st.expander("Cleaned text (NLP pipeline output)"):
                st.code(cleaned if cleaned else "(empty after cleaning)")

# ---------- Page: Model Comparison ----------
elif page == "📊 Model Comparison":
    st.markdown('<div class="fade-in"><h2>📊 Model Comparison</h2>'
                '<p style="color:#A6AEC8;">Performance of trained models on the held-out test set.</p></div>',
                unsafe_allow_html=True)

    st.dataframe(cmp_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                                     "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                 use_container_width=True)

    # ranking
    ranked = cmp_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    st.markdown("#### 🏆 Model Ranking (by F1-Score)")
    for i, row in ranked.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "•"
        st.markdown(f"""
        <div class="glass" style="margin-bottom:0.6rem; display:flex; justify-content:space-between; align-items:center;">
          <div><span style="font-size:1.4rem;">{medal}</span> <b>{row['Model']}</b></div>
          <div style="color:#A6AEC8;">Acc {row['Accuracy']:.4f} · P {row['Precision']:.4f} · R {row['Recall']:.4f} · F1 {row['F1-Score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # chart
    melted = cmp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_sequence=["#7C5CFC", "#00D4FF", "#16D9A6"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#F4F6FF", yaxis_range=[0, 1.05], height=420)
    st.plotly_chart(fig, use_container_width=True)

    # confusion matrices
    st.markdown("#### Confusion Matrices")
    cms, y_test = compute_confusion_matrices()
    cols = st.columns(3)
    for idx, (name, cm) in enumerate(cms.items()):
        with cols[idx]:
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Ham", "Spam"], y=["Ham", "Spam"],
                colorscale=[[0, "#151D35"], [1, "#7C5CFC"]],
                text=cm, texttemplate="%{text}", showscale=False))
            fig.update_layout(title=name, paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#F4F6FF", height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ---------- Page: Data Insights ----------
elif page == "📈 Data Insights":
    st.markdown('<div class="fade-in"><h2>📈 Data Insights</h2>'
                '<p style="color:#A6AEC8;">Exploratory analysis of the SMS Spam Collection dataset.</p></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(names=["Legitimate", "Spam"], values=[ham_count, spam_count],
                     color_discrete_sequence=["#16D9A6", "#FF5277"], hole=0.5)
        fig.update_layout(title="Class Distribution", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#F4F6FF", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df = load_processed()
        df["msg_len"] = df["message"].astype(str).str.len() if "message" in df.columns else df["cleaned_text"].str.len()
        fig = px.histogram(df, x="msg_len", color="label", barmode="overlay", nbins=40,
                           color_discrete_map={"ham": "#16D9A6", "spam": "#FF5277"})
        fig.update_layout(title="Message Length Distribution", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#F4F6FF", height=360)
        st.plotly_chart(fig, use_container_width=True)

    spam_counter, ham_counter = compute_word_freq()
    top_spam = pd.DataFrame(spam_counter.most_common(15), columns=["word", "count"])
    top_ham = pd.DataFrame(ham_counter.most_common(15), columns=["word", "count"])

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(top_spam, x="count", y="word", orientation="h",
                     color_discrete_sequence=["#FF5277"], title="Top Spam Words")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#F4F6FF", yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.bar(top_ham, x="count", y="word", orientation="h",
                     color_discrete_sequence=["#16D9A6"], title="Top Legitimate Words")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#F4F6FF", yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dataset Statistics")
    stats = pd.DataFrame({
        "Metric": ["Raw messages", "After deduplication", "Spam (raw)", "Ham (raw)", "Spam (processed)", "Ham (processed)"],
        "Value": [5574, total_count, 747, 4827, spam_count, ham_count],
    })
    st.dataframe(stats, use_container_width=True)

# ---------- Page: About ----------
elif page == "ℹ️ About":
    st.markdown('<div class="fade-in"><h2>ℹ️ About</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
      <h3>Project Overview</h3>
      <p style="color:#A6AEC8;">Email/SMS Spam Detection using Natural Language Processing. Classifies messages as Spam or
      Legitimate (Ham) using TF-IDF features and three ML models.</p>
      <h3>Problem Statement</h3>
      <p style="color:#A6AEC8;">Unsolicited spam messages waste time and pose security risks. This app automatically
      identifies spam using NLP and machine learning.</p>
      <h3>NLP Pipeline</h3>
      <p style="color:#A6AEC8;">Lowercasing → HTML/URL/email removal → punctuation cleanup → tokenization → stopword removal → lemmatization → TF-IDF.</p>
      <h3>Algorithms</h3>
      <p style="color:#A6AEC8;">Multinomial Naive Bayes, Logistic Regression, Linear SVM (LinearSVC).</p>
      <h3>Dataset</h3>
      <p style="color:#A6AEC8;">UCI SMS Spam Collection (5574 messages), downloaded programmatically.</p>
      <h3>Evaluation</h3>
      <p style="color:#A6AEC8;">Stratified train/test split (80/20), metrics: Accuracy, Precision, Recall, F1-Score. Best by F1: Linear SVM.</p>
      <h3>Technologies</h3>
      <p style="color:#A6AEC8;">Python, Pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib, Plotly.</p>
    </div>
    """, unsafe_allow_html=True)