import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for modern layout and soft shadows
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained KNN model safely
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# App Header
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1E293B;">🎓 Student Performance Predictor</h1>
        <p style="color: #64748B;">Enter the student details below to predict performance using your trained KNN model.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Form Layout
with st.form("prediction_form"):
    st.markdown(
        "<div class='card'><h3>Student Information</h3>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox(
            "Course", ["Science", "Commerce", "Arts", "Engineering", "Medicine"]
        )
        study_hours = st.number_input(
            "Study Hours (per day)", min_value=0.0, max_value=24.0, value=4.5
        )
        class_attendance = st.slider(
            "Class Attendance (%)", min_value=0, max_value=100, value=85
        )
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours (per night)", min_value=0.0, max_value=12.0, value=7.0
        )
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
        study_method = st.selectbox(
            "Study Method", ["Self-Study", "Online Videos", "Group Study", "Coaching"]
        )
        facility_rating = st.slider(
            "Facility Rating (1-5)", min_value=1, max_value=5, value=3
        )
        exam_difficulty = st.selectbox(
            "Exam Difficulty", ["Easy", "Medium", "Hard"]
        )

    st.markdown("</div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Predict Outcome")

# Prediction logic
if submit_button:
    # Basic numeric encoding mapping (Modify if your training pipeline used a different encoder or preprocessor pipeline)
    gender_val = 1 if gender == "Female" else 0
    internet_val = 1 if internet_access == "Yes" else 0

    course_map = {
        "Science": 0,
        "Commerce": 1,
        "Arts": 2,
        "Engineering": 3,
        "Medicine": 4,
    }
    sleep_q_map = {"Poor": 0, "Average": 1, "Good": 2}
    study_m_map = {
        "Self-Study": 0,
        "Online Videos": 1,
        "Group Study": 2,
        "Coaching": 3,
    }
    exam_d_map = {"Easy": 0, "Medium": 1, "Hard": 2}

    features = np.array(
        [
            [
                age,
                gender_val,
                course_map.get(course, 0),
                study_hours,
                class_attendance,
                internet_val,
                sleep_hours,
                sleep_q_map.get(sleep_quality, 1),
                study_m_map.get(study_method, 0),
                facility_rating,
                exam_d_map.get(exam_difficulty, 1),
            ]
        ]
    )

    try:
        prediction = model.predict(features)
        st.markdown(
            f"""
            <div class='card' style='text-align: center; background-color: #EEF2FF; border: 1px solid #C7D2FE;'>
                <h3 style='color: #4F46E5;'>Predicted Performance Score</h3>
                <h1 style='color: #312E81;'>{prediction[0]:.2f}</h1>
            </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(
            f"Prediction Error: {e}. Note: If your `knn.pkl` file contains a Scikit-Learn Pipeline that accepts raw text or string inputs directly, you can pass the raw selectbox strings instead of mapped integers."
        )
