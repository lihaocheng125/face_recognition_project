import cv2
import numpy as np
import face_recognition, face_recognition_models
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserFace
import time
import random

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
        all_actions = ['blink', 'mouth_open', 'turn_head']
        num_actions = random.randint(1, 2)
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
    all_actions = ['blink', 'mouth_open', 'turn_head']
    num_actions = random.randint(1, 2)
    actions = random.sample(all_actions, num_actions)
    return render(request, 'face_app/register.html', {'actions': actions})

def login(request):
    if request.method == 'POST':
        # 新增：随机选择一到两个动作进行活体检测
        all_actions = ['blink', 'mouth_open', 'turn_head']
        num_actions = random.randint(1, 2)
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
    all_actions = ['blink', 'mouth_open', 'turn_head']
    num_actions = random.randint(1, 2)
    actions = random.sample(all_actions, num_actions)
    return render(request, 'face_app/login.html', {'actions': actions})

def dashboard(request):
    return render(request, 'face_app/dashboard.html')