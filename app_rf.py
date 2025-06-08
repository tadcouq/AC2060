import streamlit as st
import numpy as np
import joblib

# Load mô hình đã huấn luyện
model = joblib.load('trained_rf_model.joblib')

# Giao diện người dùng
st.title('🎓 Dự đoán điểm số sinh viên')

st.markdown("""
Nhập các thông số học tập của sinh viên để dự đoán điểm số.
""")

# Input các biến đặc trưng
study_hours = st.slider('Study Hours per Week', 0, 40, 10)
online_courses = st.slider('Online Courses Completed', 0, 10, 2)
assignment_completion = st.slider('Assignment Completion Rate (%)', 0, 100, 80)
exam_score = st.slider('Exam Score (%)', 0, 100, 75)
attendance = st.slider('Attendance Rate (%)', 0, 100, 90)
edtech_use = st.selectbox('Use of Educational Tech', [0, 1])  # 0 = No, 1 = Yes

stress_category = st.selectbox('Stress Level (High/Medium/Low)', ['High', 'Medium', 'Low'])
stress_medium = 1 if stress_category == 'Medium' else 0
stress_low = 1 if stress_category == 'Low' else 0

social_media = st.slider('Time on Social Media (hrs/week)', 0, 50, 10)
sleep_hours = st.slider('Sleep Hours per Night', 0, 12, 7)

# Button để dự đoán
if st.button('Dự đoán điểm số'):
    # Tạo mảng đầu vào
    features = np.array([[study_hours, online_courses, assignment_completion,
                          exam_score, attendance, edtech_use,
                          stress_medium, stress_low, social_media, sleep_hours]])
    
    # Dự đoán
    prediction = model.predict(features)[0]
    st.success(f'Điểm số của bạn dự đoán là: {prediction:.2f}')

    # Đổi lại định dạng điểm số, với 0.0, 1.0, 2.0, 3.0, 4.0 ứng với điểm theo thang F, D, C, B, A
    if prediction < 1.0:
        grade = 'F'
    elif prediction < 2.0:
        grade = 'D'
    elif prediction < 3.0:
        grade = 'C'
    elif prediction < 3.5:
        grade = 'B'
    else:
        grade = 'A'
    st.success(f'Điểm số tương ứng với thang điểm là: {grade}')

    # credit
    st.markdown("""
    ### Dự án này được thực hiện bởi nhóm 4, 
    - Mô hình được huấn luyện bằng Random Forest Regressor.
    - Dữ liệu được sử dụng từ [đây](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style).
    - Mô hình được huấn luyện và lưu trữ trong file `trained_rf_model.joblib`.
    - Ứng dụng được xây dựng bằng Streamlit.
    (c) 2025 https://tdcq.me/
    """)    