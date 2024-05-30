import time, cv2, os
import numpy as np

# 카메라 초기화 및 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 카메라 해상도 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 카메라 해상도 높이 설정

# 이미지 캡처 및 표시
ret, image = cap.read()  # ret: 이미지 읽기 성공 여부, image: 캡처된 이미지
if ret == True:
    cv2.imshow('CAMERA', image)  # 캡처된 이미지 창에 표시
    img_captured = cv2.imwrite('test.jpg', image)  # 이미지 파일로 저장
    resize_img = cv2.resize(image, (1435, 800))  # 이미지 크기 조정
    cv2.imshow('CAMERA', resize_img)  # 조정된 이미지 창에 표시
    cv2.waitKey(3500)  # 3.5초 대기 (밀리초 단위)

# 자원 해제
cap.release()  # 카메라 자원 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

time.sleep(1)  # 1초 대기 
