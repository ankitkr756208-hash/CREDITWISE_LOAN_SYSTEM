import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

model = pickle.load(open("model.pkl","rb"))

st.set_page_config(page_title="CreditWise Bank AI", layout="wide")

# ----------- PREMIUM BANK CSS -----------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

body {
    background: linear-gradient(135deg,#16222A,#3A6073);
}

.main {
    background: transparent;
}

.card {
    background: rgba(255,255,255,0.12);
    padding:25px;
    border-radius:18px;
    backdrop-filter: blur(14px);
    transition:0.3s;
    box-shadow:0 8px 32px rgba(0,0,0,0.2);
}
.card:hover {
    transform: translateY(-6px);
    box-shadow:0 12px 40px rgba(0,0,0,0.35);
}

.kpi {
    font-size:30px;
    font-weight:700;
    color:#00e0ff;
}

.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    border:none;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
    transition:0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow:0 0 20px #00c6ff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏦 CreditWise Banking Intelligence")

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Loan Application")

data = {}
data["Applicant_Income"] = st.sidebar.number_input("Applicant Income",0)
data["Coapplicant_Income"] = st.sidebar.number_input("Coapplicant Income",0)
data["Age"] = st.sidebar.slider("Age",18,70)
data["Dependents"] = st.sidebar.selectbox("Dependents",["0","1","2","3+"])
data["Credit_Score"] = st.sidebar.slider("Credit Score",300,900)
data["Existing_Loans"] = st.sidebar.number_input("Existing Loans",0)
data["DTI_Ratio"] = st.sidebar.slider("DTI Ratio",0.0,1.0)
data["Savings"] = st.sidebar.number_input("Savings",0)
data["Collateral_Value"] = st.sidebar.number_input("Collateral Value",0)
data["Loan_Amount"] = st.sidebar.number_input("Loan Amount",0)
data["Loan_Term"] = st.sidebar.number_input("Loan Term",0)

data["Education_Level"] = st.sidebar.selectbox("Education",["Graduate","Not Graduate"])
data["Employment_Status"] = st.sidebar.selectbox("Employment",["Salaried","Self-employed","Unemployed"])
data["Marital_Status"] = st.sidebar.selectbox("Marital",["Single","Married"])
data["Loan_Purpose"] = st.sidebar.selectbox("Purpose",["Car","Education","Home","Personal"])
data["Property_Area"] = st.sidebar.selectbox("Property",["Urban","Semiurban","Rural"])
data["Gender"] = st.sidebar.selectbox("Gender",["Male","Female"])
data["Employer_Category"] = st.sidebar.selectbox("Employer Category",["Government","MNC","Private","Unemployed"])

# ---------- KPI CARDS ----------
c1,c2,c3,c4 = st.columns(4)

c1.markdown(f"<div class='card'><div class='kpi'>₹{data['Applicant_Income']}</div>Applicant Income</div>",unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='kpi'>₹{data['Loan_Amount']}</div>Loan Amount</div>",unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='kpi'>{data['Credit_Score']}</div>Credit Score</div>",unsafe_allow_html=True)
c4.markdown(f"<div class='card'><div class='kpi'>{data['Age']}</div>Age</div>",unsafe_allow_html=True)

# ---------- GRAPHICAL BAR ----------
bar_df = pd.DataFrame({
    "Metric":["Income","Loan","Savings","Collateral"],
    "Value":[data["Applicant_Income"],data["Loan_Amount"],data["Savings"],data["Collateral_Value"]]
})

fig_bar = px.bar(bar_df, x="Metric", y="Value", color="Metric",
                 color_discrete_sequence=["#00c6ff","#0072ff","#4facfe","#43e97b"])

st.plotly_chart(fig_bar,use_container_width=True)

# ---------- PREDICTION ----------
if st.button("🚀 Analyze Loan Application"):

    df = pd.DataFrame([data])

    pred = model.predict(df)

    classes = model.named_steps['classifier'].classes_
    prob = model.predict_proba(df)[0][list(classes).index('Yes')]

    # Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob*100,
        title = {'text': "Approval Probability"},
        gauge = {
            'axis': {'range': [0,100]},
            'bar': {'color': "#00e0ff"},
            'steps' : [
                {'range': [0,40], 'color': "#ff5f6d"},
                {'range': [40,70], 'color': "#ffc371"},
                {'range': [70,100], 'color': "#00f260"}],
        }))

    st.plotly_chart(fig, use_container_width=True)

    if pred[0] == "Yes":
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")