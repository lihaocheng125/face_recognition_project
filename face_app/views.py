import cv2
import numpy as np
import face_recognition, face_recognition_models
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserFace
import time
import random
import math

# 新增：张嘴检测函数
def detect_mouth_open(frame):
    # 这里可以使用更复杂的方法，例如使用面部关键点检测来判断嘴是否张开
    # 简单示例：检测面部轮廓，判断嘴部区域的变化
    face_landmarks_list = face_recognition.face_landmarks(frame)
    if face_landmarks_list:
        face_landmarks = face_landmarks_list[0]
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']
        # 计算上下唇之间的平均距离
        lip_distances = [np.linalg.norm(np.array(top_lip[i]) - np.array(bottom_lip[i])) for i in range(len(top_lip))]
        avg_lip_distance = sum(lip_distances) / len(lip_distances)
        # 假设一个阈值，大于该阈值认为嘴张开
        if avg_lip_distance > 20:
            return True
    return False

# 新增：转头检测函数
def detect_turn_head(frame):
    # 这里可以使用更复杂的方法，例如使用头部姿态估计来判断是否转头
    # 简单示例：检测面部的左右位置变化
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_x = (left + right) / 2
        frame_center_x = frame.shape[1] / 2
        # 假设一个阈值，面部中心与帧中心的距离大于该阈值认为转头
        if abs(face_center_x - frame_center_x) > 50:
            return True
    return False

# 新增：点头检测函数
def detect_nod(frame, prev_face_center_y=None):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_y = (top + bottom) / 2
        if prev_face_center_y is not None:
            # 假设角度变化转换为像素距离阈值（需要根据实际情况调整）
            y_diff = abs(face_center_y - prev_face_center_y)
            if y_diff > 20:  # 假设这个像素差值对应 15° - 30° 角度变化
                return True
    return False

# 新增：摇头检测函数
def detect_shake_head(frame, prev_face_center_x=None):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_x = (left + right) / 2
        if prev_face_center_x is not None:
            # 假设角度变化转换为像素距离阈值（需要根据实际情况调整）
            x_diff = abs(face_center_x - prev_face_center_x)
            if x_diff > 30:  # 假设这个像素差值对应 30° - 45° 角度变化
                return True
    return False

# 新增：侧头检测函数
def detect_tilt_head(frame):
    face_landmarks_list = face_recognition.face_landmarks(frame)
    if face_landmarks_list:
        face_landmarks = face_landmarks_list[0]
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        # 计算两只眼睛连线的角度（简单示例，需要更精确的算法）
        angle = math.atan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
        if abs(angle) > 0.35:  # 假设这个角度对应 20° - 30° 侧头角度
            return True
    return False

# 新增：抬头检测函数
def detect_lift_head(frame, prev_face_center_y=None):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_y = (top + bottom) / 2
        if prev_face_center_y is not None:
            # 假设角度变化转换为像素距离阈值（需要根据实际情况调整）
            y_diff = face_center_y - prev_face_center_y
            if y_diff < -20:  # 假设这个像素差值对应 15° - 25° 抬头角度
                return True
    return False

# 新增：低头检测函数
def detect_lower_head(frame, prev_face_center_y=None):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_center_y = (top + bottom) / 2
        if prev_face_center_y is not None:
            # 假设角度变化转换为像素距离阈值（需要根据实际情况调整）
            y_diff = face_center_y - prev_face_center_y
            if y_diff > 20:  # 假设这个像素差值对应 15° - 20° 低头角度
                return True
    return False

# 新增：挥手检测函数（简单示例，需要更复杂的手势识别算法）
def detect_wave_hand(frame, prev_hand_center=None):
    # 这里假设使用简单的颜色检测来识别手部（需要根据实际情况调整）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if prev_hand_center is not None:
                # 假设速度转换为像素距离阈值（需要根据实际情况调整）
                dist = math.sqrt((cX - prev_hand_center[0])**2 + (cY - prev_hand_center[1])**2)
                if dist > 20:  # 假设这个像素差值对应 0.5 - 1 米/秒 的速度
                    return True
    return False

# 新增：举手检测函数（简单示例，需要更复杂的姿态识别算法）
def detect_raise_hand(frame):
    # 这里假设使用简单的颜色检测来识别手部（需要根据实际情况调整）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])
            if cY < frame.shape[0] / 2:  # 假设手部在画面上半部分认为是举手
                return True
    return False

def detect_blinks(frame, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    return len(eyes)

def live_detection(actions):
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    start_time = time.time()
    blink_count = 0
    blink_threshold = 2  # 眨眼次数阈值
    time_limit = 5  # 时间区间（秒）
    mouth_open_detected = False
    head_turn_detected = False
    nod_detected = False
    shake_head_detected = False
    tilt_head_detected = False
    lift_head_detected = False
    lower_head_detected = False
    wave_hand_detected = False
    raise_hand_detected = False

    for action in actions:
        if action == 'blink':
            blink_start_time = time.time()
            while time.time() - blink_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                eye_count = detect_blinks(frame, eye_cascade)
                if eye_count < 2:
                    blink_count += 1
            if blink_count >= blink_threshold:
                blink_success = True
            else:
                blink_success = False
        elif action == 'mouth_open':
            mouth_open_start_time = time.time()
            while time.time() - mouth_open_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                if detect_mouth_open(frame):
                    mouth_open_detected = True
            mouth_open_success = mouth_open_detected
        elif action == 'turn_head':
            head_turn_start_time = time.time()
            while time.time() - head_turn_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                if detect_turn_head(frame):
                    head_turn_detected = True
            head_turn_success = head_turn_detected
        elif action == 'nod':
            # 这里添加点头检测逻辑
            nod_start_time = time.time()
            while time.time() - nod_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测头部上下位置变化
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_center_y = (top + bottom) / 2
                    prev_face_center_y = face_center_y
                    for _ in range(10):  # 检测一段时间内的变化
                        ret, frame = cap.read()
                        if not ret:
                            break
                        face_locations = face_recognition.face_locations(frame)
                        if face_locations:
                            top, right, bottom, left = face_locations[0]
                            face_center_y = (top + bottom) / 2
                            if abs(face_center_y - prev_face_center_y) > 20:
                                nod_detected = True
                                break
            nod_success = nod_detected
        elif action == 'shake_head':
            # 这里添加摇头检测逻辑
            shake_head_start_time = time.time()
            while time.time() - shake_head_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测头部左右位置变化
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_center_x = (left + right) / 2
                    prev_face_center_x = face_center_x
                    for _ in range(10):  # 检测一段时间内的变化
                        ret, frame = cap.read()
                        if not ret:
                            break
                        face_locations = face_recognition.face_locations(frame)
                        if face_locations:
                            top, right, bottom, left = face_locations[0]
                            face_center_x = (left + right) / 2
                            if abs(face_center_x - prev_face_center_x) > 20:
                                shake_head_detected = True
                                break
            shake_head_success = shake_head_detected
        elif action == 'tilt_head':
            # 这里添加歪头检测逻辑
            tilt_head_start_time = time.time()
            while time.time() - tilt_head_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测头部倾斜角度变化
                face_landmarks_list = face_recognition.face_landmarks(frame)
                if face_landmarks_list:
                    face_landmarks = face_landmarks_list[0]
                    left_eye = face_landmarks['left_eye']
                    right_eye = face_landmarks['right_eye']
                    left_eye_center = np.mean(np.array(left_eye), axis=0)
                    right_eye_center = np.mean(np.array(right_eye), axis=0)
                    eye_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
                    prev_eye_angle = eye_angle
                    for _ in range(10):  # 检测一段时间内的变化
                        ret, frame = cap.read()
                        if not ret:
                            break
                        face_landmarks_list = face_recognition.face_landmarks(frame)
                        if face_landmarks_list:
                            face_landmarks = face_landmarks_list[0]
                            left_eye = face_landmarks['left_eye']
                            right_eye = face_landmarks['right_eye']
                            left_eye_center = np.mean(np.array(left_eye), axis=0)
                            right_eye_center = np.mean(np.array(right_eye), axis=0)
                            eye_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
                            if abs(eye_angle - prev_eye_angle) > 0.2:
                                tilt_head_detected = True
                                break
            tilt_head_success = tilt_head_detected
        elif action == 'lift_head':
            # 这里添加抬头检测逻辑
            lift_head_start_time = time.time()
            while time.time() - lift_head_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测头部上下位置变化
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_center_y = (top + bottom) / 2
                    prev_face_center_y = face_center_y
                    for _ in range(10):  # 检测一段时间内的变化
                        ret, frame = cap.read()
                        if not ret:
                            break
                        face_locations = face_recognition.face_locations(frame)
                        if face_locations:
                            top, right, bottom, left = face_locations[0]
                            face_center_y = (top + bottom) / 2
                            if face_center_y < prev_face_center_y - 20:
                                lift_head_detected = True
                                break
            lift_head_success = lift_head_detected
        elif action == 'lower_head':
            # 这里添加低头检测逻辑
            lower_head_start_time = time.time()
            while time.time() - lower_head_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测头部上下位置变化
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_center_y = (top + bottom) / 2
                    prev_face_center_y = face_center_y
                    for _ in range(10):  # 检测一段时间内的变化
                        ret, frame = cap.read()
                        if not ret:
                            break
                        face_locations = face_recognition.face_locations(frame)
                        if face_locations:
                            top, right, bottom, left = face_locations[0]
                            face_center_y = (top + bottom) / 2
                            if face_center_y > prev_face_center_y + 20:
                                lower_head_detected = True
                                break
            lower_head_success = lower_head_detected
        elif action == 'wave_hand':
            # 这里添加挥手检测逻辑
            wave_hand_start_time = time.time()
            while time.time() - wave_hand_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测手部位置变化
                # 需要使用更复杂的手部检测方法，这里只是示例
                pass
            wave_hand_success = wave_hand_detected
        elif action == 'raise_hand':
            # 这里添加举手检测逻辑
            raise_hand_start_time = time.time()
            while time.time() - raise_hand_start_time < time_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # 示例：简单检测手部位置变化
                # 需要使用更复杂的手部检测方法，这里只是示例
                pass
            raise_hand_success = raise_hand_detected

    cap.release()

    if 'blink' in actions:
        if not blink_success:
            return False
    if 'mouth_open' in actions:
        if not mouth_open_success:
            return False
    if 'turn_head' in actions:
        if not head_turn_success:
            return False
    if 'nod' in actions:
        if not nod_success:
            return False
    if 'shake_head' in actions:
        if not shake_head_success:
            return False
    if 'tilt_head' in actions:
        if not tilt_head_success:
            return False
    if 'lift_head' in actions:
        if not lift_head_success:
            return False
    if 'lower_head' in actions:
        if not lower_head_success:
            return False
    if 'wave_hand' in actions:
        if not wave_hand_success:
            return False
    if 'raise_hand' in actions:
        if not raise_hand_success:
            return False

    return True
def index(request):
    return render(request, 'face_app/index.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        if UserFace.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('register')

        # 新增：多个活体检测动作
        all_actions = ['blink', 'mouth_open', 'turn_head', 'nod', 'shake_head', 'tilt_head', 'lift_head', 'lower_head', 'wave_hand', 'raise_hand']
        num_actions = 1  # 固定为 1，确保每次只做一个动作
        actions = random.sample(all_actions, num_actions)

        if not live_detection(actions):
            messages.error(request, 'Live detection failed. Please try again.')
            return redirect('register')

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            messages.error(request, 'No face detected. Please try again.')
            return redirect('register')

        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        face_encoding_str = ','.join(map(str, face_encoding))
        UserFace.objects.create(username=username, face_encoding=face_encoding_str)
        messages.success(request, 'Registration successful. You can now log in.')
        return redirect('index')

    # 随机选择动作并传递到模板
    all_actions = ['blink', 'mouth_open', 'turn_head', 'nod', 'shake_head', 'tilt_head', 'lift_head', 'lower_head', 'wave_hand', 'raise_hand']
    num_actions = 1  # 固定为 1，确保每次只做一个动作
    actions = random.sample(all_actions, num_actions)
    return render(request, 'face_app/register.html', {'actions': actions})

def login(request):
    if request.method == 'POST':
        # 新增：随机选择一到两个动作进行活体检测
        all_actions = ['blink', 'mouth_open', 'turn_head', 'nod', 'shake_head', 'tilt_head', 'lift_head', 'lower_head', 'wave_hand', 'raise_hand']
        num_actions = 1  # 固定为 1，确保每次只做一个动作
        actions = random.sample(all_actions, num_actions)

        if not live_detection(actions):
            messages.error(request, 'Live detection failed. Please try again.')
            return redirect('login')

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            messages.error(request, 'No face detected. Please try again.')
            return redirect('login')

        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        for user_face in UserFace.objects.all():
            stored_encoding = np.fromstring(user_face.face_encoding, sep=',')
            results = face_recognition.compare_faces([stored_encoding], face_encoding)
            if results[0]:
                messages.success(request, f'Welcome, {user_face.username}!')
                return redirect('dashboard')

        messages.error(request, 'Login failed. No matching user found.')
        return redirect('login')

    # 随机选择动作并传递到模板
    all_actions = ['blink', 'mouth_open', 'turn_head', 'nod', 'shake_head', 'tilt_head', 'lift_head', 'lower_head', 'wave_hand', 'raise_hand']
    num_actions = 1  # 固定为 1，确保每次只做一个动作
    actions = random.sample(all_actions, num_actions)
    return render(request, 'face_app/login.html', {'actions': actions})
def dashboard(request):
    return render(request, 'face_app/dashboard.html')