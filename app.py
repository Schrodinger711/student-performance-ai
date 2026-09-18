import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("student_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance AI")
st.write(
    "Predict a student's final grade using previous academic performance "
    "and selected student information."
)

st.divider()

# Input section
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=25,
        value=17
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        index=1,
        help="1 = <2 hours, 2 = 2-5 hours, 3 = 5-10 hours, 4 = >10 hours"
    )

    failures = st.number_input(
        "Past Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

with col2:
    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5
    )

    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    G2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

st.divider()

# Prediction button
if st.button("🎯 Predict Final Grade", use_container_width=True):

    input_data = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2]
    })

    prediction = model.predict(input_data)[0]

    # Keep prediction within valid grade range
    prediction = max(0, min(20, prediction))

    st.subheader("📊 Prediction")

    # Display prediction
    st.metric(
        label="Predicted Final Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )

    # Interpretation
    if prediction >= 16:
        st.success("🌟 Excellent predicted performance!")
    elif prediction >= 12:
        st.success("👍 Good predicted performance.")
    elif prediction >= 10:
        st.info("📚 Predicted performance is around the passing range.")
    else:
        st.warning("⚠️ The model predicts a lower final grade.")

    # Show input data
    with st.expander("View student data used for prediction"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

st.divider()

st.caption(
    "Built using Python, Pandas, Scikit-learn, Random Forest and Streamlit."
)