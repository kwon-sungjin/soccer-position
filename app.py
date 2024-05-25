import requests
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import joblib
import os

def download_model(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        st.success(f"Model downloaded successfully: {filename}")
    else:
        st.error(f"Failed to download model. Status code: {response.status_code}")
# 모델 파일 URL과 로컬 저장 파일명
model_url = "https://github.com/your-username/your-repo/raw/main/final_model.h5"
model_filename = "final_model.h5"

# 모델 파일 다운로드
download_model(model_url, model_filename)

# 모델 및 scaler 로드
if os.path.exists(model_filename):
    model = load_model(model_filename)
    st.success("Model loaded successfully.")
else:
    st.error("Model file not found.")
scaler = joblib.load('scaler.pkl')

# 포지션 매핑 및 조언
position_mapping = {
    0: 'CAM', 1: 'CB', 2: 'CDM', 3: 'ST', 4: 'CM',
    5: 'GK', 6: 'SB', 7: 'SM'
}

position_advice = {
    'CAM': "As a Central Attacking Midfielder, focus on improving your vision, passing accuracy, and creativity to create goal-scoring opportunities.",
    'CB': "As a Center Back, work on your tackling, aerial ability, and positioning to defend against opposing attackers effectively.",
    'CDM': "As a Central Defensive Midfielder, enhance your ball interception skills, stamina, and ability to read the game to protect your defense.",
    'ST': "As a Striker, concentrate on your finishing, positioning, and pace to be in the right place at the right time to score goals.",
    'CM': "As a Central Midfielder, develop your passing range, vision, and ability to control the tempo of the game.",
    'GK': "As a Goalkeeper, focus on your shot-stopping, command of the area, and distribution to help your team from the back.",
    'SB': "As a Side Back, improve your crossing, tackling, and stamina to support both defense and attack along the flanks.",
    'SM': "As a Side Midfielder, work on your dribbling, crossing, and pace to create chances from wide areas."
}

# MTCNN 인스턴스 생성
detector = MTCNN()

# 새로운 데이터 전처리 함수
def preprocess_new_data(image, height, weight):
    # 얼굴 추출
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # ResNet50 입력 크기로 조정
        face_img = np.array(face_img) / 255.0  # 정규화
        face_img = np.expand_dims(face_img, axis=0)  # 배치 차원 추가
    else:
        st.warning("No face detected in the uploaded image.")
        face_img = np.zeros((1, 224, 224, 3))  # 기본값으로 빈 이미지 사용

    # 키와 몸무게 데이터 (정규화)
    numeric_data = np.array([[height, weight]])
    numeric_data = scaler.transform(numeric_data)  # 정규화 적용

    return face_img, numeric_data

# Streamlit 앱 인터페이스
st.title('Football Player Position Prediction')

uploaded_file = st.file_uploader("Upload an image of the player", type=["jpg", "jpeg", "png"])
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=180)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=150, value=70)

if st.button('Predict Position'):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        face_img, numeric_data = preprocess_new_data(image, height, weight)

        predictions = model.predict([face_img, numeric_data])
        predicted_position = np.argmax(predictions, axis=1)

        predicted_position_name = position_mapping[predicted_position[0]]
        st.write(f"Predicted position: {predicted_position_name}")
        st.write(position_advice[predicted_position_name])

        # 예측 확률 출력 (추가)
        probabilities = predictions[0]
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{position_mapping[i]}: {prob * 100:.2f}%")
    else:
        st.warning("Please upload an image.")
