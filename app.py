import streamlit as st
import joblib
import numpy as np

st.title("Employee Salary Prediction App")

st.divider()


st.write("This app predicts the salary of an employee based on their years of experiencel.")

years = st.number_input("Enter Years of Experience", min_value=0, step = 1, value=1)

jobrate = st.number_input("Enter Job Rating", min_value=0.0, step=0.5, value=3.0)

x = [years, jobrate]

model = joblib.load("salary_prediction_model.pkl")
st.divider()

predict_button = st.button("Predict Salary")

st.divider()

if predict_button:

    st.balloons()

    X1 = np.array([x])

    predict_button = model.predict(X1)

    st.write(f"The predicted salary is: {predict_button[0]}")
    st.sidebar.text("1. Enter the years of experience and job rating.")
    st.sidebar.text("2. Click on 'Predict Salary' to get the estimated salary.")
    st.sidebar.text("3. Enjoy the balloons!")
    st.sidebar.text("4. The model is trained using a linear regression algorithm.")
    st.sidebar.text("5. The model is saved as 'salary_prediction_model.pkl'.")
    st.sidebar.text("6. The app is built using Streamlit.")
    st.sidebar.text("7. The app is designed to be user-friendly and interactive.")
    st.sidebar.text("8. The app is deployed on a local server.")
    st.sidebar.text("9. The app is designed to be responsive and easy to use.")
    st.sidebar.text("10. The app is designed to be visually appealing and easy to navigate.")
    