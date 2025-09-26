import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# --- Caching model training ---
@st.cache_data
def train_model():
    """Train a logistic regression model with sample data"""
    X_train = pd.DataFrame({
        'Age': [20, 21, 19, 22, 23, 18, 24, 20],
        'GPA': [3.2, 2.5, 3.8, 2.0, 2.7, 3.5, 1.8, 2.9],
        'Attendance': [85, 70, 90, 60, 75, 95, 50, 80]
    })
    y_train = [0, 1, 0, 1, 1, 0, 1, 0]  # 0 = No dropout, 1 = Dropout
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

@st.cache_data
def predict_dropout(age, gpa, attendance):
    """Predict dropout probability for a single student"""
    model = train_model()
    X_new = pd.DataFrame({'Age': [age], 'GPA': [gpa], 'Attendance': [attendance]})
    proba = model.predict_proba(X_new)[0][1]  # Probability of dropout
    return proba

# --- Initialize or load data in session state ---
def init_data():
    if "students" not in st.session_state:
        st.session_state.students = pd.DataFrame({
            "Student": ["Alice", "Bob", "Charlie", "Diana"],
            "Age": [20, 21, 19, 22],
            "GPA": [3.2, 2.5, 3.8, 2.0],
            "Attendance": [85, 70, 90, 60],
            "Dropout Probability": [0.2, 0.7, 0.4, 0.9]
        })

# --- Streamlit App ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Input", "Prediction Results"])

init_data()

# --- HOME PAGE ---
if page == "Home":
    st.title("Dropout Prediction App")
    st.write("""
        Welcome! Use the sidebar to input student data and see predictions.
        This app predicts dropout probability based on **Age**, **GPA**, and **Attendance**.
    """)
    st.write("### Sample Data")
    st.dataframe(st.session_state.students)

# --- DATA INPUT PAGE ---
elif page == "Data Input":
    st.title("Enter Student Data (Live Prediction)")

    name = st.text_input("Student Name")
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gpa = st.slider("GPA", 0.0, 4.0, 2.5, 0.1)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)

    if name.strip():
        # Predict dropout
        pred = predict_dropout(age, gpa, attendance)
        st.write(f"Predicted dropout probability for **{name.strip()}**: **{pred:.2%}**")

        # Add to dataset
        if st.button("Add this student to the dataset"):
            new_row_df = pd.DataFrame([{
                "Student": name.strip(),
                "Age": age,
                "GPA": gpa,
                "Attendance": attendance,
                "Dropout Probability": pred
            }])

            # ✅ Use pd.concat instead of append
            st.session_state.students = pd.concat(
                [st.session_state.students, new_row_df],
                ignore_index=True
            )
            st.success(f"Student {name.strip()} added successfully!")
    else:
        st.info("Please enter the student's name to see prediction.")

# --- PREDICTION RESULTS PAGE ---
elif page == "Prediction Results":
    st.title("Prediction Results & Visualization")
    data = st.session_state.students.copy()

    # Visualization
    fig = px.bar(
        data,
        x="Student",
        y="Dropout Probability",
        color="Dropout Probability",
        color_continuous_scale="Viridis",
        range_y=[0, 1],
        title="Dropout Probability per Student"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional stats
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Summary Stats")
        st.write(data.describe())
    with col2:


