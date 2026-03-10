import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="HealthGuard",
    page_icon="nurse.png",
    layout="wide"
)

# =============================
# SESSION STATE
# =============================

if "menu" not in st.session_state:
    st.session_state.menu = "Home"

if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================
# MODERN CSS
# =============================

st.markdown("""
<style>

[data-testid="stSidebar"]{
background-color:#f4f8ff;
}

.sidebar-title{
font-size:28px;
font-weight:bold;
text-align:center;
color:#ff4b8b;
margin-bottom:20px;
}

.title{
text-align:center;
font-size:65px;
font-weight:900;
color:#ff4b8b;
}

.subtitle{
text-align:center;
font-size:22px;
color:gray;
margin-bottom:30px;
}

.card{
background:white;
padding:40px;
border-radius:20px;
text-align:center;
font-size:20px;
font-weight:bold;
box-shadow:0 10px 25px rgba(0,0,0,0.1);
transition:0.3s;
}

.card:hover{
transform:translateY(-10px);
box-shadow:0 15px 35px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATASET
# =============================

diabetes = pd.read_csv("diabetes.csv")
heart = pd.read_csv("heart_v2(in).csv")

# =============================
# TRAIN MODEL DIABETES
# =============================

features_diabetes = [
'Pregnancies',
'Insulin',
'BMI',
'Age',
'Glucose',
'BloodPressure',
'DiabetesPedigreeFunction'
]

X = diabetes[features_diabetes]
y = diabetes["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.3,random_state=1
)

model_rf_diabetes = RandomForestClassifier()
model_rf_diabetes.fit(X_train,y_train)

model_dt_diabetes = DecisionTreeClassifier()
model_dt_diabetes.fit(X_train,y_train)

rf_acc_diabetes = accuracy_score(y_test, model_rf_diabetes.predict(X_test))
dt_acc_diabetes = accuracy_score(y_test, model_dt_diabetes.predict(X_test))

# =============================
# TRAIN MODEL HEART
# =============================

Xh = heart.drop("heart disease",axis=1)
yh = heart["heart disease"]

X_train_h,X_test_h,y_train_h,y_test_h = train_test_split(
Xh,yh,test_size=0.3,random_state=1
)

model_rf_heart = RandomForestClassifier()
model_rf_heart.fit(X_train_h,y_train_h)

model_dt_heart = DecisionTreeClassifier()
model_dt_heart.fit(X_train_h,y_train_h)

rf_acc_heart = accuracy_score(y_test_h, model_rf_heart.predict(X_test_h))
dt_acc_heart = accuracy_score(y_test_h, model_dt_heart.predict(X_test_h))

# =============================
# SIDEBAR
# =============================

st.sidebar.markdown("<div class='sidebar-title'>🧑‍⚕️ HealthGuard</div>", unsafe_allow_html=True)

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.menu = "Home"

if st.sidebar.button("🩸 Prediksi Diabetes", use_container_width=True):
    st.session_state.menu = "Prediksi Diabetes"

if st.sidebar.button("❤️ Prediksi Jantung", use_container_width=True):
    st.session_state.menu = "Prediksi Jantung"

menu = st.session_state.menu

# =============================
# HOME PAGE
# =============================

if menu == "Home":

    if st.session_state.page == "home":

        st.markdown("<div class='title'>HealthGuard</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>AI Medical Prediction System</div>", unsafe_allow_html=True)

        logo = "nurse.png"

        col1,col2,col3 = st.columns([2,3,2])

        with col2:
            if os.path.exists(logo):
                st.image(logo,width="stretch")

        st.write("")
        st.write("")

        col1,col2 = st.columns(2)

        with col1:
            if st.button("🩸 Informasi Diabetes",use_container_width=True):
                st.session_state.page = "diabetes_info"
                st.rerun()
                
        with col2:
            if st.button("❤️ Informasi Penyakit Jantung",use_container_width=True):
                st.session_state.page = "heart_info"
                st.rerun()
                
    if st.session_state.page == "diabetes_info":

        st.title("🩸 Informasi Lengkap Diabetes")

        st.subheader("Apa itu Diabetes")
        st.write("""
Diabetes adalah penyakit kronis yang terjadi ketika kadar gula darah terlalu tinggi karena tubuh tidak dapat memproduksi atau menggunakan insulin dengan baik.
""")
        st.subheader("Penyebab")
        st.write("""
        • Faktor genetik  
        • Pola makan tinggi gula  
        • Obesitas  
        • Kurang aktivitas fisik
        """)

        st.subheader("Gejala")
        st.write("""
        • Sering haus  
        •   Sering buang air kecil  
        • Mudah lelah  
        • Penurunan berat badan
        """)

        st.subheader("Pencegahan")
        st.write("""
        • Pola makan sehat  
        • Olahraga rutin  
        • Mengurangi konsumsi gula  
        • Menjaga berat badan
        """)
        if st.button("⬅ Kembali ke Home"):
            st.session_state.page = "home"

    if st.session_state.page == "heart_info":

        st.title("❤️ Informasi Penyakit Jantung")

        st.subheader("Apa itu Penyakit Jantung")
        st.write("""
Penyakit jantung adalah kondisi yang mempengaruhi fungsi jantung dan pembuluh darah sehingga aliran darah tidak berjalan normal.
""")

        st.subheader("Penyebab")
        st.write("""
        • Kolesterol tinggi  
        • Tekanan darah tinggi  
        • Merokok  
        • Obesitas
        """)

        st.subheader("Gejala")
        st.write("""
        • Nyeri dada  
        • Sesak napas  
        • Detak jantung tidak teratur
        """)

        st.subheader("Pencegahan")
        st.write("""
        • Pola hidup sehat  
        • Olahraga rutin  
        • Menghindari rokok  
        • Mengontrol tekanan darah
        """)

        if st.button("⬅ Kembali ke Home"):
            st.session_state.page = "home"

# =============================
# PREDIKSI DIABETES
# =============================

elif menu == "Prediksi Diabetes":

    st.title("🩸 Prediksi Diabetes")

    st.write("Random Forest Accuracy :", round(rf_acc_diabetes*100,2), "%")
    st.write("Decision Tree Accuracy :", round(dt_acc_diabetes*100,2), "%")

    col1,col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies (times)",step=1)
        bmi = st.number_input("BMI (kg/m²)")
        glucose = st.number_input("Glucose (mg/dL)")
        dpf = st.number_input("Diabetes Pedigree Function")

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)")
        age = st.number_input("Age (years)",step=1)
        bp = st.number_input("Blood Pressure (mmHg)")

    if st.button("🔍 Predict Diabetes"):

        with st.spinner("AI sedang menganalisis data pasien..."):
            time.sleep(2)

            data = pd.DataFrame([[preg,insulin,bmi,age,glucose,bp,dpf]],
            columns=features_diabetes)

            pred_rf = model_rf_diabetes.predict(data)
            prob = model_rf_diabetes.predict_proba(data)
            pred_dt = model_dt_diabetes.predict(data)

            risk = prob[0][1]*100

        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text':"Risk %"},
        gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Random Forest")
        if pred_rf[0] == 1:
            st.error("⚠ Terindikasi Diabetes")
        else:
            st.success("✅ Tidak Terindikasi Diabetes")

        st.subheader("Decision Tree")
        if pred_dt[0] == 1:
            st.error("⚠ Terindikasi Diabetes")
        else:
            st.success("✅ Tidak Terindikasi Diabetes")

# =============================
# PREDIKSI JANTUNG
# =============================

elif menu == "Prediksi Jantung":

    st.title("❤️ Prediksi Penyakit Jantung")

    st.write("Random Forest Accuracy :", round(rf_acc_heart*100,2), "%")
    st.write("Decision Tree Accuracy :", round(dt_acc_heart*100,2), "%")

    col1,col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)",step=1)
        bp = st.number_input("Blood Pressure (mmHg)")

    with col2:
        gender = st.selectbox("Gender",["Female","Male"])
        chol = st.number_input("Cholesterol (mg/dL)")

    sex = 1 if gender=="Male" else 0

    if st.button("🔍 Predict Heart Disease"):

        with st.spinner("AI sedang menganalisis data pasien..."):
            time.sleep(2)

            data = pd.DataFrame([[age,sex,bp,chol]],columns=Xh.columns)

            pred_rf = model_rf_heart.predict(data)
            prob = model_rf_heart.predict_proba(data)
            pred_dt = model_dt_heart.predict(data)

            risk = prob[0][1]*100

        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text':"Risk %"},
        gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Random Forest")
        if pred_rf[0] == 1:
            st.error("⚠ Terindikasi Penyakit Jantung")
        else:
            st.success("✅ Tidak Ada Indikasi")

        st.subheader("Decision Tree")
        if pred_dt[0] == 1:
            st.error("⚠ Terindikasi Penyakit Jantung")
        else:

            st.success("✅ Tidak Ada Indikasi")

