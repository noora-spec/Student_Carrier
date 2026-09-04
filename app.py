import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Placement Prediction App", layout="wide", page_icon="🎓")

# Load & Train Model dengan Cache
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('student_career_success_dataset.csv')
    
    # Hapus kolom identitas & data leakage
    drop_cols = ['Student_ID', 'Company_Tier', 'Career_Field', 'Placement_Mode', 'Starting_Salary_USD']
    X = df.drop(columns=drop_cols + ['Placement_Status'])
    y = df['Placement_Status']

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Encode Categorical Features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Fit Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'acc': accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred),
        'rec': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return model, le, X_encoded.columns.tolist(), metrics, X_test, y_test, y_proba, df

model, le, feature_columns, metrics, X_test, y_test, y_proba, df = load_and_train_model()

# Header Utama
st.title("🎓 Student Placement Prediction App")
st.write("Aplikasi prediktif kelulusan dan penempatan kerja mahasiswa berbasis **Random Forest Classifier**.")

# Tab Layanan
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Kelulusan", "📊 Evaluasi Model", "📁 Dataset"])

# TAB 1: PREDIKSI
with tab1:
    st.subheader("Masukkan Parameter Mahasiswa")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Umur", 18, 35, 22)
        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female', 'Other'])
        year = st.selectbox("Tahun Angkatan", ['Freshman', 'Sophomore', 'Junior', 'Senior'])
        major = st.selectbox("Jurusan", sorted(df['Major'].unique()))
        academic_perf = st.selectbox("Kinerja Akademik", ['Poor', 'Average', 'Good', 'Excellent'])
        cgpa = st.slider("IPK / CGPA", 0.0, 4.0, 3.2, step=0.01)

    with col2:
        attendance = st.slider("Kehadiran Kuliah (%)", 0, 100, 85)
        study_hours = st.number_input("Jam Belajar / Minggu", 0, 80, 20)
        prog_skill = st.slider("Skill Pemrograman (1-10)", 1, 10, 7)
        projects = st.number_input("Jumlah Proyek Selesai", 0, 30, 5)
        certs = st.number_input("Sertifikasi Dimiliki", 0, 20, 2)
        internships = st.number_input("Pengalaman Magang", 0, 10, 1)

    with col3:
        interview_score = st.slider("Nilai Interview (0-100)", 0, 100, 75)
        employability_score = st.slider("Skor Employability", 0.0, 300.0, 200.0)
        resume_score = st.slider("Skor Resume (0-100)", 0, 100, 80)
        github = st.selectbox("Profil GitHub Active?", ['Yes', 'No'])
        linkedin = st.selectbox("Profil LinkedIn Active?", ['Yes', 'No'])
        leadership = st.selectbox("Pengalaman Organisasi?", ['Yes', 'No'])
        english = st.selectbox("Kemampuan Bahasa Inggris", ['Basic', 'Intermediate', 'Advanced'])

    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        input_data = {
            'Age': age, 'Attendance_Percentage': attendance, 'Study_Hours_Per_Week': study_hours,
            'CGPA': cgpa, 'Programming_Skill': prog_skill, 'Projects_Completed': projects,
            'Certifications': certs, 'Hackathons': 0, 'Internships': internships,
            'Resume_Score': resume_score, 'Communication_Skills': 7, 'Teamwork': 7,
            'Problem_Solving': 7, 'Interview_Score': interview_score,
            'Employability_Score': employability_score, 'Gender': gender,
            'University_Year': year, 'Major': major, 'Academic_Performance': academic_perf,
            'GitHub_Profile': github, 'Leadership_Experience': leadership,
            'LinkedIn_Profile': linkedin, 'English_Proficiency': english
        }

        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]
        result_label = le.inverse_transform([prediction])[0]

        st.divider()
        if result_label == 'Placed':
            st.success(f"### 🎉 Status Prediksi: **Placed** (Peluang: {probabilities[1]*100:.1f}%)")
        else:
            st.error(f"### ⚠️ Status Prediksi: **Not Placed** (Peluang: {probabilities[0]*100:.1f}%)")

# TAB 2: METRIK & VISUALISASI
with tab2:
    st.subheader("Metrik Performa Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['acc']*100:.2f}%")
    m2.metric("Precision", f"{metrics['prec']*100:.2f}%")
    m3.metric("Recall", f"{metrics['rec']*100:.2f}%")
    m4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

    with col_right:
        st.markdown("**Top 10 Feature Importances**")
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        feat_imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(10)
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_fi, palette='viridis')
        ax_fi.set_xlabel('Importance Score')
        st.pyplot(fig_fi)

# TAB 3: DATA OVERVIEW
with tab3:
    st.subheader("Sampel Data")
    st.dataframe(df.head(100), use_container_width=True)
