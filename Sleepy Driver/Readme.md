😴 Drowsiness Detection Program 🚙
============================

Project Overview 🖥️
---------------------------
### 개발 환경
  > window10, python3.10, pycharm23.2.5, teachable machine
### 개발 목적
  > 졸음운전으로 인한 교통사고는 2017년에 2002건, 2018년에 1308건, 2019년에 2887건 발생하였습니다. 음주운전과 다르게 졸음운전은 자신의 의지만으로는 이겨내기 어렵다고 생각했습니다. 그러나 졸음쉼터 또는 휴게소에 잠시 휴식을 취하는 것만으로도 사고 예방이 가능합니다. 그래서 졸음의 기준을 스스로 판단하는 것이 아니라 프로그램에서 경고하면 휴식 필요성을 인지하고 행동할 수 있는 프로그램을 개발하고 싶었습니다.   
![졸음운전 교통사고](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/1a4db201-f1a7-457b-8794-c8e22843d510)   
(공공데이터 포털에서 제공된 경찰청 사고 현황 데이터, https://www.data.go.kr/data/15047952/fileData.do)
### 기능
  > 캠을 통해서 눈을 감은 상태 카운트가 15가 될 경우 경고음과 함께 전국 휴게소와 졸음쉼터 위치가 표시된 지도를 출력합니다. 지도에는 휴게소와 졸음쉼터를 구별하여 확인할 수 있는 체크박스가 있으며, 위치 클릭시 화장실, 음식점, 편의시설에 대한 부가적인 정보가 있는 팝업 창이 표시됩니다.
   
-----------------------------
   
✨ Project Description ✨
-----------------------------
+ 폴더 및 파일 설명
  - data_set   
    : teachable machine에서 close_samples과 open_samples을 각각 1500개의 이미지로 학습시킨 데이터 폴더.
  - rest area data   
    : 지도에 표시되는 휴게소와 졸음쉼터 데이터 파일로 공공데이터 포털에서 다운로드 받음.   
  - Drowsiness_Detection.py   
    : 프로그램이 진행되는 python source code.   
  - alarm.wav   
    : 눈을 15초 이상 감아 졸음으로 감지하면 울리는 경고음   
  - keras_model.h5   
    : teachable machine에서 학습 후 모델 변환하여 다운로드 받은 opencv_keras model.   
  - korea.html   
    : 휴게소와 졸음쉼터가 표시되는 지도 html.
  - labels.txt   
    : keras model에 대한 label로 0은 open eyes, 1은 close eyes를 뜻함.
    
+  개발 과정 설명   
   - teachable machine 모델 설명   
     학습 후 제공되는 keras_model.h5, labels.txt, source code 활용
    ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/578abd74-db71-45a1-88b6-38b12ecc5b36)   
    ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/86088bde-b10c-43e7-a4fb-86ae7cde54f4)   

   - <Drowsiness_Detection.py> 코드 설명   
    ``` python
    import pandas as pd
    import folium
    import webbrowser
    from keras.models import load_model
    import cv2
    import numpy as np
    import winsound
    import time
    ```
    > pip install and import library 필요   
    > pandas는 csv 파일 읽기, folium은 지도 마킹, webbrowser는 지도 저장 및 출력, load_model은 keras.model 읽기, cv2는 pip install opencv-python 필요하며 웹캠 사용, numpy는 모델 연산, winsound는 경고음 출력, time은 count 누적에 대한 시간 초 부여   
    ```
    sleep_count = 0
    ```
    > sleep_count 변수 처음에 0으로 초기화
    ```
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
    ```
    > keras.model 사용에 대한 코드
    
    ```
    # close 인식할 때마다 sleep_count 1 증가
    if class_name == class_names[1]:
      sleep_count += 1
      time.sleep(0.1)
      print (sleep_count)
    # open 인식하면 sleep_count 0으로 초기화
    else:
      sleep_count = 0
    ```
    > 캠에서 close로 인식하면 sleep_count 1씩 증가하며 open으로 인식하면 count 0으로 초기화
    ```
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
    ```
    > sleep_count가 15가 되면 데이터 파일 읽어온다.   
    > folium 라이브러리 통해 지도에 휴게소와 졸음쉬터 위치 및 부가정보를 표시   
    > green mark는 졸음쉼터, red mark는 휴게소로 표시   
    ```
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
    ```
    > 졸음쉼터와 휴게소를 분리하여 볼 수 있도록 체크박스 생성   
    > korea.html에 데이터 저장 및 출력   
    > 졸음 감지 경고음 출력 후 sleep_count 0으로 초기화   
    ```
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

    camera.release()
    cv2.destroyAllWindows()
    ```
    > 프로그램 종료 위한 코드
+ Program 실행 설명
  ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/389e7e3a-f09d-41b0-aa0e-7f3528358216)
  ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/d90f1fc5-b73e-44b4-84a9-4a0c1555e8e0)
  ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/fdacdbf7-fc8c-4cf5-b22a-6d2be2923d61)
  ![image](https://github.com/Jeonsowon/Jeonsowon/assets/144200709/bb3badd5-d778-4169-a83b-7fc1ad861787)
  > 한국만 데이터 적용   
-----------------------------
Reference 🙏
-----------------------------
+ https://blog.naver.com/ysjang0102/223021863973 (Teachable machine 이용한 코드 구상 참조)
+ https://www.data.go.kr/data/15028203/standard.do (졸음쉼터 데이터 다운로드)
+ https://www.data.go.kr/data/15025446/standard.do (휴게소 데이터 다운로드)

Developer 🙋
-----------------------------
+ 21102318 전소원
