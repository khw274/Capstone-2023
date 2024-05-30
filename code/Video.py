if '77차1004' in chars:  # '77차1004'가 입력된 경우
    print('          * 인식 성공 *')  # '인식 성공' 메시지를 출력

    file = open('77차1004.txt', mode='r')  # '77차1004.txt' 파일을 읽기 모드로 오픈
    print(file.read()) 

    Vid = cv2.VideoCapture('car1.mp4')  # 미리 저장해둔 'car1.mp4' 비디오 파일을 읽어옴

    # 비디오가 열렸는지 확인하고, 속성들을 변수에 저장합니다.
    if Vid.isOpened():  
        fps = Vid.get(cv2.CAP_PROP_FPS)  # 프레임 속도
        f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)  # 프레임 수
        f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # 프레임 너비
        f_height = Vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 프레임 높이

    # 비디오가 열려있는 동안 실행
    while Vid.isOpened():
        ret, frame = Vid.read()  # 비디오에서 프레임을 읽어옵니다.
        if ret:  # 프레임을 성공적으로 읽어왔다면,
            # 프레임 크기를 조정함
            re_frame = cv2.resize(frame, (round(f_width / 1.33), round(f_height / 1.35)))
            # 조정된 프레임을 화면에 표시합니다.
            cv2.imshow('Car_Video', re_frame)
            key = cv2.waitKey(10)  # 키 입력을 대기하며, 10밀리초마다 프레임을 업데이트합니다.

            if key == ord('q'):  # 'q' 키가 눌리면,
                break  # 비디오 재생을 멈춥니다.
        else:
            break  # 프레임을 더 이상 읽어올 수 없으면 비디오 재생을 멈춥니다.
    Vid.release()  # 비디오 장치를 닫습니다.
    cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.

elif '96쇼2962' in chars:  # 두 번째 차
    print('          * 인식 성공 *')

    file = open('96쇼2962.txt', mode = 'r')
    print(file.read())

    Vid = cv2.VideoCapture('car2.mp4')  # 두 번째 차 인터페이스

    if Vid.isOpened():
        fps = Vid. get(cv2.CAP_PROP_FPS)
        f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
        f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_height = Vid.get(cv2. CAP_PROP_FRAME_HEIGHT)

    while Vid.isOpened() :
        ret, frame = Vid. read() 
        if ret:
            re_frame = cv2.resize(frame, (round(f_width/1.33), round(f_height/1.35)))
            cv2.imshow('Car_Video',re_frame)
            key = cv2.waitKey(10)

            if key == ord('q'):
                break
            
        else:
            break
    Vid.release()
    cv2.destroyAllWindows()

elif '86타8558' in chars:  # 세 번째 차
    print('          * 인식 성공 *')

    file = open('86타8558.txt', mode = 'r')
    print(file.read())

    Vid = cv2.VideoCapture('car3.mp4')  # 세 번째 차 인터페이스

    if Vid.isOpened():
        fps = Vid. get(cv2.CAP_PROP_FPS)
        f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
        f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_height = Vid.get(cv2. CAP_PROP_FRAME_HEIGHT)

    while Vid.isOpened() :
        ret, frame = Vid. read() 
        if ret:
            re_frame = cv2.resize(frame, (round(f_width/1.33), round(f_height/1.35)))
            cv2.imshow('Car_Video',re_frame)
            key = cv2.waitKey(10)

            if key == ord('q'):
                break
            
        else:
            break
    Vid.release()
    cv2.destroyAllWindows()
    
else:
    print('인식 실패')  # 인식 실패시 '인식 실패' 를 출력

time.sleep(1)
