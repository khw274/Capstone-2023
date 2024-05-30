import RPi.GPIO as GPIO  # 라즈베리파이 GPIO 제어 라이브러리

GPIO.setmode(GPIO.BCM)  # GPIO 핀 번호 체계를 BCM 모드로 설정
GPIO.setup(18, GPIO.IN, GPIO.PUD_UP)  # GPIO 18번 핀을 입력 모드로 설정하고 풀업 저항 활성화

while 1:  # 무한 루프 시작
    x = GPIO.input(18)  # GPIO 18번 핀의 입력 값을 읽어 변수 x에 저장
    # print(x)  # 디버깅용: 스위치가 눌리면 x = 0, 눌리지 않으면 x = 1

    if x == 0:  # 스위치가 눌렸을 때
        print('스위치 ON')  # '스위치 ON' 메시지 출력

        # 이후 코드들 실행
