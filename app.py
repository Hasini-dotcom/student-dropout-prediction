# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('model/dropout_model.pkl')
    encoders = joblib.load('model/label_encoders.pkl')
    features = joblib.load('model/feature_names.pkl')
    explainer = joblib.load('model/shap_explainer.pkl')
    return model, encoders, features, explainer

@st.cache_data
def load_data():
    df = pd.read_csv('data/student_dropout_dataset.csv')
    return df

model, encoders, features, explainer = load_model()
df_raw = load_data()

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def encode_row(row_dict):
    text_cols = ['gender', 'state', 'family_income_category', 'parents_education']
    for col in text_cols:
        le = encoders[col]
        val = row_dict[col]
        if val in le.classes_:
            row_dict[col] = le.transform([val])[0]
        else:
            row_dict[col] = 0
    return row_dict

def predict_student(row_dict):
    encoded = encode_row(row_dict.copy())
    input_df = pd.DataFrame([encoded])[features]
    prob = model.predict_proba(input_df)[0][1]
    if prob >= 0.65:
        label = 'HIGH'
        color = '#FF4B4B'
        emoji = '🔴'
    elif prob >= 0.35:
        label = 'MEDIUM'
        color = '#FFA500'
        emoji = '🟡'
    else:
        label = 'LOW'
        color = '#00C853'
        emoji = '🟢'
    return prob, label, color, emoji, input_df

def get_interventions(label, row):
    interventions = []
    if label in ['HIGH', 'MEDIUM']:
        if row.get('current_attendance', 100) < 60:
            interventions.append("📋 Schedule immediate attendance counselling session")
        if row.get('child_labour_involved', 0) == 1:
            interventions.append("🏛️ Connect family to Child Labour Rehabilitation scheme")
        if row.get('socioeconomic_score', 10) < 4:
            interventions.append("💰 Refer family to PM POSHAN & scholarship programs")
        if row.get('distance_from_school_km', 0) > 8:
            interventions.append("🚌 Apply for government transport/hostel facility")
        if row.get('midterm_score', 100) < 40:
            interventions.append("📚 Enrol student in remedial coaching programme")
        if row.get('has_elder_sibling_dropped', 0) == 1:
            interventions.append("👨‍👩‍👧 Conduct family awareness session on education importance")
        if row.get('health_issues', 0) == 1:
            interventions.append("🏥 Refer to school health programme / medical checkup")
        if not interventions:
            interventions.append("👁️ Monitor student closely over next 30 days")
            interventions.append("📞 Schedule parent-teacher meeting")
    else:
        interventions.append("✅ Student is on track — continue regular monitoring")
    return interventions

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("🎓 Dropout Predictor")
st.sidebar.markdown("AI-Powered Early Warning System for Government Schools")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔍 Predict Single Student",
    "📊 School Analytics",
    "📁 Batch Prediction",
    "ℹ️ About"
])

# ─────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("🎓 AI-Powered Student Dropout Prediction System")
    st.markdown("#### For Government Schools Across India")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", "2,000")
    col2.metric("High Risk Students", str(len(df_raw[df_raw['dropout_risk_label'] == 'High'])))
    col3.metric("Model Accuracy", "93%")
    col4.metric("States Covered", "8")

    st.markdown("---")
    st.subheader("📌 Risk Distribution Across All Students")

    risk_counts = df_raw['dropout_risk_label'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {'High': '#FF4B4B', 'Medium': '#FFA500', 'Low': '#00C853'}
    bars = ax.bar(risk_counts.index, risk_counts.values,
                  color=[colors.get(x, 'gray') for x in risk_counts.index])
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Students")
    ax.set_title("Students by Dropout Risk Level")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("⚠️ High Risk Students — Quick View")
    high_risk = df_raw[df_raw['dropout_risk_label'] == 'High'][[
        'student_id', 'class', 'gender', 'state',
        'current_attendance', 'midterm_score', 'dropout_risk_score'
    ]].head(10)
    high_risk['dropout_risk_score'] = high_risk['dropout_risk_score'].round(1)
    st.dataframe(high_risk, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — PREDICT SINGLE STUDENT
# ─────────────────────────────────────────
elif page == "🔍 Predict Single Student":
    st.title("🔍 Predict Dropout Risk for a Student")
    st.markdown("Fill in the student's details below:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Basic Info**")
        student_name = st.text_input("Student Name", "Ravi Kumar")
        student_class = st.selectbox("Class", [6, 7, 8, 9, 10])
        gender = st.selectbox("Gender", ["Male", "Female"])
        state = st.selectbox("State", ['Telangana', 'UP', 'Bihar', 'Rajasthan', 'MP', 'Odisha', 'Jharkhand', 'Assam'])

    with col2:
        st.markdown("**🏠 Socioeconomic**")
        socioeconomic_score = st.slider("Socioeconomic Score (0=Poor, 10=Rich)", 0.0, 10.0, 2.5)
        family_income = st.selectbox("Family Income", ['Below Poverty Line', 'Low Income', 'Middle Income'])
        distance = st.slider("Distance from School (km)", 0.5, 25.0, 5.0)
        num_siblings = st.slider("Number of Siblings", 0, 5, 2)
        parents_edu = st.selectbox("Parents Education", ['Illiterate', 'Primary', 'Secondary', 'Graduate'])
        elder_sibling_dropped = st.selectbox("Elder Sibling Dropped Out?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_first_gen = st.selectbox("First Generation Learner?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        st.markdown("**📚 Academic & Risk Factors**")
        prev_attendance = st.slider("Previous Year Attendance (%)", 10.0, 100.0, 75.0)
        curr_attendance = st.slider("Current Attendance (%)", 10.0, 100.0, 60.0)
        midterm = st.slider("Midterm Score (%)", 0.0, 100.0, 45.0)
        final_prev = st.slider("Previous Year Final Score (%)", 0.0, 100.0, 50.0)
        failed_prev = st.selectbox("Failed Previous Year?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        repeated = st.selectbox("Repeated Class?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        child_labour = st.selectbox("Child Labour Involved?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        migration = st.selectbox("Seasonal Migration?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        health = st.selectbox("Health Issues?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        teacher_ratio = st.slider("Teacher-Student Ratio", 30, 65, 50)
        midday_meal = st.selectbox("School Has Midday Meal?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        toilet = st.selectbox("School Has Toilet?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("---")

    if st.button("🔮 Predict Dropout Risk", use_container_width=True):
        row = {
            'class': student_class, 'gender': gender, 'state': state,
            'socioeconomic_score': socioeconomic_score,
            'family_income_category': family_income,
            'distance_from_school_km': distance,
            'num_siblings': num_siblings, 'parents_education': parents_edu,
            'is_first_gen_learner': is_first_gen,
            'has_elder_sibling_dropped': elder_sibling_dropped,
            'child_labour_involved': child_labour,
            'seasonal_migration': migration, 'health_issues': health,
            'prev_year_attendance': prev_attendance,
            'current_attendance': curr_attendance,
            'midterm_score': midterm, 'final_score_prev_year': final_prev,
            'score_trend': midterm - final_prev,
            'failed_prev_year': failed_prev, 'repeated_class': repeated,
            'teacher_student_ratio': teacher_ratio,
            'school_has_midday_meal': midday_meal,
            'school_has_toilet': toilet
        }

        prob, label, color, emoji, input_df = predict_student(row)

        st.markdown(f"""
        <div style='background-color:{color}20; border-left: 6px solid {color};
        padding: 20px; border-radius: 10px; margin: 10px 0'>
        <h2 style='color:{color}'>{emoji} {student_name} — {label} RISK</h2>
        <h3 style='color:{color}'>Dropout Probability: {prob*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Recommended Interventions")
        for tip in get_interventions(label, row):
            st.markdown(f"- {tip}")

        st.markdown("### 🔬 Why is this student at risk? (SHAP Explanation)")
        shap_vals = explainer.shap_values(input_df)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap_series = pd.Series(shap_vals[0], index=features).abs().sort_values(ascending=True).tail(8)
        colors_shap = ['#FF4B4B' if v > 0 else '#00C853'
                       for v in pd.Series(shap_vals[0], index=features).loc[shap_series.index]]
        shap_series.plot(kind='barh', ax=ax, color=colors_shap)
        ax.set_title(f"Top Risk Factors for {student_name}")
        ax.set_xlabel("Impact on Risk Score")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────
# PAGE 3 — SCHOOL ANALYTICS
# ─────────────────────────────────────────
elif page == "📊 School Analytics":
    st.title("📊 School-Level Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk by Class")
        class_risk = df_raw.groupby(['class', 'dropout_risk_label']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        class_risk.plot(kind='bar', ax=ax,
                        color={'High': '#FF4B4B', 'Medium': '#FFA500', 'Low': '#00C853'})
        ax.set_xlabel("Class")
        ax.set_ylabel("Students")
        ax.set_title("Dropout Risk by Class")
        plt.xticks(rotation=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk by State")
        state_high = df_raw[df_raw['dropout_risk_label'] == 'High']['state'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        state_high.plot(kind='bar', ax=ax, color='#FF4B4B')
        ax.set_xlabel("State")
        ax.set_ylabel("High Risk Students")
        ax.set_title("High Risk Students by State")
        plt.xticks(rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Attendance vs Risk")
        fig, ax = plt.subplots(figsize=(6, 4))
        for risk, color in [('High', '#FF4B4B'), ('Medium', '#FFA500'), ('Low', '#00C853')]:
            subset = df_raw[df_raw['dropout_risk_label'] == risk]['current_attendance']
            ax.hist(subset, bins=20, alpha=0.6, label=risk, color=color)
        ax.set_xlabel("Current Attendance (%)")
        ax.set_ylabel("Number of Students")
        ax.set_title("Attendance Distribution by Risk")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Key Stats")
        high = df_raw[df_raw['dropout_risk_label'] == 'High']
        st.metric("Avg Attendance (High Risk)", f"{high['current_attendance'].mean():.1f}%")
        st.metric("Avg Midterm Score (High Risk)", f"{high['midterm_score'].mean():.1f}%")
        st.metric("Child Labour in High Risk", f"{high['child_labour_involved'].mean()*100:.1f}%")
        st.metric("Elder Sibling Dropped (High Risk)", f"{high['has_elder_sibling_dropped'].mean()*100:.1f}%")

# ─────────────────────────────────────────
# PAGE 4 — BATCH PREDICTION
# ─────────────────────────────────────────
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction — Full Class/School")
    st.markdown("Upload your school's student data CSV to get risk scores for all students at once.")
    st.markdown("---")

    if st.button("📊 Run on Sample Dataset", use_container_width=True):
        sample = df_raw.head(50).copy()
        text_cols = ['gender', 'state', 'family_income_category', 'parents_education']
        encoded = sample.copy()
        for col in text_cols:
            le = encoders[col]
            encoded[col] = le.transform(encoded[col])
        X_sample = encoded[features]
        probs = model.predict_proba(X_sample)[:, 1]
        sample['Predicted_Probability'] = (probs * 100).round(1)
        sample['Predicted_Risk'] = pd.cut(probs,
            bins=[0, 0.35, 0.65, 1.0],
            labels=['🟢 LOW', '🟡 MEDIUM', '🔴 HIGH'])
        display_cols = ['student_id', 'class', 'gender', 'state',
                        'current_attendance', 'midterm_score',
                        'Predicted_Probability', 'Predicted_Risk']
        st.dataframe(sample[display_cols], use_container_width=True)
        st.success(f"✅ Processed 50 students. High risk: {(probs >= 0.65).sum()}")

# ─────────────────────────────────────────
# PAGE 5 — ABOUT
# ─────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### AI-Powered Student Dropout Prediction System
    **Bharat Academix CodeQuest Hackathon 2026**

    ---
    ### 👥 Team
    | Role | Name |
    |------|------|
    | Team Leader | Sudham Hasini |
    | Team Member | Pillalamarri Harshitha |
    | Team Member | Pillalamarri Swastika |

    ---
    ### 🛠️ Tech Stack
    - **ML Model:** XGBoost Classifier
    - **Explainability:** SHAP (SHapley Values)
    - **Dashboard:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualisation:** Matplotlib, Seaborn

    ---
    ### 📊 Model Performance
    - **Accuracy:** 93%
    - **Dropout Precision:** 81%
    - **Training Data:** 2,000 students across 8 Indian states

    ---
    ### 🎯 Impact
    Even a 10% dropout reduction means **15 lakh additional students**
    completing secondary education annually across India.
    """)