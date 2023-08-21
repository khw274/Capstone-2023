import RPi.GPIO as GPIO 
import time, cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pytesseract

GPIO.setmode(GPIO.BCM)  # new
GPIO.setup(18, GPIO.IN, GPIO.PUD_UP)  # new
while 1:
    x = GPIO.input(18) 
    # print(x) #누르면 x=1, 아닐 시 0

    if x == 0:
        print('스위치 ON')

        #카메라 동작 코딩
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

        ret, image = cap.read() #ret(이미지 읽는 성공 여부)
        if ret == True:
            cv2.imshow('CAMERA',image)
            img_captured = cv2.imwrite('test.jpg',image)
            resize_img = cv2.resize(image, (1435, 800))
            cv2.imshow('CAMERA',resize_img)
            cv2.waitKey(3500)


        cap.release()
        cv2.destroyAllWindows()

        time.sleep(1)
        plt.style.use('dark_background')

        img_ori = cv2.imread('test.jpg') #이미지 불러옴

        height, width, channel = img_ori.shape #너비 높이 채널 저장

        # plt.figure(figsize=(12, 10))
        # plt.imshow(img_ori, cmap='gray')
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) #이미지 그레이

        # plt.figure(figsize=(12, 10))
        # plt.imshow(gray, cmap='gray') #gray image

        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #노이즈 제거, 블러 처리

        img_thresh = cv2.adaptiveThreshold( #이진화 이미지/픽셀 높으면 255(흰), 낮으면 0(검)
            img_blurred, 
            maxValue=255.0, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=19, 
            C=9
        )

        # plt.figure(figsize=(12, 10))
        # plt.imshow(img_thresh, cmap='gray')

        contours,hierarchy = cv2.findContours( #윤곽선 그리기
            img_thresh, 
            mode=cv2.RETR_LIST, 
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
        # -1 = 전체 컨투어를 그림 

        # plt.figure(figsize=(12, 10))
        # plt.imshow(temp_result)

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        contours_dict = [] #컨투어 정보 저장

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour) #컨투어의 사각형 범위 추출
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
            

            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        # plt.figure(figsize=(12, 10))
        # plt.imshow(temp_result, cmap='gray')

        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        possible_contours = [] #가능한 숫자들 저장

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h'] 
            
            if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
                
        # visualize possible contours
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

        # plt.figure(figsize=(12, 10))
        # plt.imshow(temp_result, cmap='gray')

        MAX_DIAG_MULTIPLYER = 5 # 5
        MAX_ANGLE_DIFF = 12.0 # 12.0
        MAX_AREA_DIFF = 0.5 # 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3 # 3

        def find_chars(contour_list):
            matched_result_idx = []
            
            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])

                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))
                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                # append this contour
                matched_contours_idx.append(d1['idx'])

                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
                
                # recursive
                recursive_contour_list = find_chars(unmatched_contour)
                
                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx
            
        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))

        # visualize possible contours
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for r in matched_result:
            for d in r:
        #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

        # plt.figure(figsize=(12, 10))
        # plt.imshow(temp_result, cmap='gray')

        PLATE_WIDTH_PADDING = 1.3 # 1.3
        PLATE_HEIGHT_PADDING = 1.5 # 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []

        for i, matched_chars in enumerate(matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
            
            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
            
            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
            
            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )
            
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
            
            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
            
            img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
            
            img_cropped = cv2.getRectSubPix(
                img_rotated, 
                patchSize=(int(plate_width), int(plate_height)), 
                center=(int(plate_cx), int(plate_cy))
            )
            
            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue
            
            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })
            
            # plt.subplot(len(matched_result), 1, i+1)
            # plt.imshow(img_cropped, cmap='gray')


        longest_idx, longest_text = -1, 0
        plate_chars = []

        for i, plate_img in enumerate(plate_imgs):
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # find contours again (same as above)
            contours, hierarchy = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
            
            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                area = w * h
                ratio = w / h

                if area > MIN_AREA \
                and w > MIN_WIDTH and h > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
                    if x < plate_min_x:
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h
                        
            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
            
            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

            chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
            
            result_chars = ''
            has_digit = False
            for c in chars:
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                    if c.isdigit():
                        has_digit = True
                    result_chars += c
            
            # print(result_chars)
            plate_chars.append(result_chars)

            if has_digit and len(result_chars) > longest_text:
                longest_idx = i

            # plt.subplot(len(plate_imgs), 1, i+1)
            # plt.imshow(img_result, cmap='gray')

            info = plate_infos[longest_idx]
        chars = plate_chars[longest_idx]

        # print("추출된 번호 : ", chars)

        img_out = img_ori.copy()

        cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

        cv2.imwrite(chars + ' 번호 영역.jpg', img_out)
        cv2.imwrite(chars + ' 번호 추출.jpg', img_cropped)

        # plt.figure(figsize=(12, 10))
        # plt.imshow(img_out)
        # plt.show()

        if '0' in chars: #첫 번째 차
            print('          * 인식 성공 *')
            # x = img.imread('success.jpg')

            # fig = plt.figure(figsize=(14.4, 8.2), facecolor='white')
            # fig.patch.set_alpha(1) #투명도
            # plt.imshow(x)
            # plt.axis('off')
            # plt.show()

            file = open('77차1004.txt', mode = 'r')
            print(file.read())

            Vid = cv2.VideoCapture('car1.mp4') #첫 번째 차 인터페이스

            if Vid.isOpened():
                fps = Vid. get(cv2.CAP_PROP_FPS)
                f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
                f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                f_height = Vid.get(cv2. CAP_PROP_FRAME_HEIGHT)

            # print('Frames per second : ', fps, 'FPS')
            # print('Frame count : ', f_count) 
            # print('Frame width : ', f_width)
            # print('Frame height : ', f_height)

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

        elif '2' in chars: #두 번째 차
            print('          * 인식 성공 *')
            # x = img.imread('success.jpg')

            # fig = plt.figure(figsize=(14.4, 8.2), facecolor='white')
            # fig.patch.set_alpha(1) #투명도
            # plt.imshow(x)
            # plt.axis('off')
            # plt.show()

            file = open('96쇼2962.txt', mode = 'r')
            print(file.read())

            Vid = cv2.VideoCapture('car2.mp4') #두 번째 차 인터페이스

            if Vid.isOpened():
                fps = Vid. get(cv2.CAP_PROP_FPS)
                f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
                f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                f_height = Vid.get(cv2. CAP_PROP_FRAME_HEIGHT)

            # print('Frames per second : ', fps, 'FPS')
            # print('Frame count : ', f_count) 
            # print('Frame width : ', f_width)
            # print('Frame height : ', f_height)

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

        elif '8' in chars: #세 번째 차
            print('          * 인식 성공 *')
            # x = img.imread('success.jpg')

            # fig = plt.figure(figsize=(14.4, 8.2), facecolor='white')
            # fig.patch.set_alpha(1) #투명도
            # plt.imshow(x)
            # plt.axis('off')
            # plt.show()

            file = open('86타8558.txt', mode = 'r')
            print(file.read())

            Vid = cv2.VideoCapture('car3.mp4') #세 번째 차 인터페이스

            if Vid.isOpened():
                fps = Vid. get(cv2.CAP_PROP_FPS)
                f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
                f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                f_height = Vid.get(cv2. CAP_PROP_FRAME_HEIGHT)

            # print('Frames per second : ', fps, 'FPS')
            # print('Frame count : ', f_count) 
            # print('Frame width : ', f_width)
            # print('Frame height : ', f_height)

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
            print('인식 실패')

        time.sleep(1)
        # 파일 삭제(초기화)
        file = 'test.jpg'
        if os.path.exists(file):
            os.remove(file)
            # print('파일삭제 완료')


        break