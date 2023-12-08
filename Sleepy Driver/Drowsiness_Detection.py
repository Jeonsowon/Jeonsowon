import pandas as pd
import folium
import webbrowser
from keras.models import load_model
import cv2
import numpy as np
import winsound
import time

sleep_count = 0

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224))

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    # close 인식할 때마다 sleep_count 1 증가
    if class_name == class_names[1]:
        sleep_count += 1
        time.sleep(0.1)
        print (sleep_count)
    # open 인식하면 sleep_count 0으로 초기화
    else:
        sleep_count = 0
    # sleep_count 15까지 누적되면 졸음쉼터와 휴게소 정보있는 지도와 알람 출력
    if sleep_count == 15:
        rest = pd.read_csv('전국졸음쉼터표준데이터.csv', encoding='cp949')
        area = pd.read_csv('전국휴게소정보표준데이터.csv', encoding='cp949')

        # 지도 기준 위치 '서울과학기술대학교'
        m = folium.Map(
            location=[float(37.63186), float(127.0775)],
            zoom_start=10
        )

        tooltip = 'Click'

        # 졸음쉼터와 휴게소 그룹설정하여 분리
        졸음쉼터 = folium.FeatureGroup(name='졸음쉼터').add_to(m)
        휴게소 = folium.FeatureGroup(name='휴게소').add_to(m)

        # 졸음쉼터 data 지도에 마킹
        for i in range(rest.shape[0]):
            folium.Marker(
                [rest.iloc[i]['위도'], rest.iloc[i]['경도']],
                popup=f'<div style="width:200px"><strong>졸음쉼터명: {rest.iloc[i]["졸음쉼터명"]}</strong><br>'
                      f'<br>'
                      f'도로노선명: {rest.iloc[i]["도로노선명"]}<br>'
                      f'도로노선방향: {rest.iloc[i]["도로노선방향"]}<br>'
                      f'화장실: {rest.iloc[i]["화장실유무"]}<br> </div >',
                tooltip=tooltip,
                icon=folium.Icon(color='green', icon='star')
            ).add_to(졸음쉼터)

        # 휴게소 data 지도에 마킹
        for i in range(area.shape[0]):
            folium.Marker(
                [area.iloc[i]['위도'], area.iloc[i]['경도']],
                popup=f'<div style="width:300px"><strong>휴게소명: {area.iloc[i]["휴게소명"]}</strong><br>'
                      f'<br>'
                      f'도로노선명: {area.iloc[i]["도로노선명"]}<br>'
                      f'도로노선방향: {area.iloc[i]["도로노선방향"]}<br>'
                      f'휴게소 운영시간: {area.iloc[i]["휴게소운영시작시각"]} ~ {area.iloc[i]["휴게소운영종료시각"]}<br>'
                      f'화장실: {area.iloc[i]["화장실유무"]}<br>'
                      f'경정비 가능: {area.iloc[i]["경정비가능여부"]}<br>'
                      f'주유소: {area.iloc[i]["주유소유무"]} | LPG 충전소: {area.iloc[i]["LPG충전소유무"]} | 전기차 충전소: {area.iloc[i]["전기차충전소유무"]}<br>'
                      f'편의시설: 약국{area.iloc[i]["약국유무"]} 수유실{area.iloc[i]["수유실유무"]} 매점{area.iloc[i]["매점유무"]}<br>'
                      f'음식점: {area.iloc[i]["음식점유무"]} (대표음식 : {area.iloc[i]["휴게소대표음식명"]})<br>'
                      f'</div >',
                tooltip=tooltip,
                icon=folium.Icon(color='red', icon='star')
            ).add_to(휴게소)

        # 졸음쉼터, 휴게소 그룹 체크박스
        folium.LayerControl(collapsed=False).add_to(m)

        # korea 지도 html에 저장 및 출력
        m.save('korea.html')
        filepath = "korea.html"
        webbrowser.open_new_tab(filepath)

        # 졸음 감지로 인한 경고음 출력
        winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)

        # sleep_count 0으로 초기화
        sleep_count = 0

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
