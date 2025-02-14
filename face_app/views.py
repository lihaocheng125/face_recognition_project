import cv2
import numpy as np
import face_recognition,face_recognition_models
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserFace
import time

def detect_blinks(frame, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    return len(eyes)

def live_detection():
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    start_time = time.time()
    blink_count = 0
    blink_threshold = 2  # 眨眼次数阈值
    time_limit = 5  # 时间区间（秒）

    while time.time() - start_time < time_limit:
        ret, frame = cap.read()
        if not ret:
            break
        eye_count = detect_blinks(frame, eye_cascade)
        if eye_count < 2:
            blink_count += 1
    cap.release()
    return blink_count >= blink_threshold

def index(request):
    return render(request, 'face_app/index.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        if UserFace.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('register')

        if not live_detection():
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

    return render(request, 'face_app/register.html')

def login(request):
    if request.method == 'POST':
        if not live_detection():
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

    return render(request, 'face_app/login.html')

def dashboard(request):
    return render(request, 'face_app/dashboard.html')