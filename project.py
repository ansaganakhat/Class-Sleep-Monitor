"""
import cv2
import mediapipe as mp

# Инициализация Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk

# ----------------------
# Настройки
# ----------------------
EAR_THRESHOLD = 0.25       # порог закрытого глаза
CLOSED_FRAMES = 30         # количество кадров для "спящего" (примерно 3 сек при 10 fps)
MAX_FACES = 20

# ----------------------
# Инициализация MediaPipe Face Mesh
# ----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------
# Индексы глаз (MediaPipe 468 landmarks)
# ----------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_indices])
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ----------------------
# Tkinter окно для отображения спящих учеников
# ----------------------
root = tk.Tk()
root.title("Спящие ученики")
label = tk.Label(root, text="Спят: 0/20", font=("Arial", 24))
label.pack()

# ----------------------
# Камера
# ----------------------
cap = cv2.VideoCapture(0)

# Словарь для хранения количества кадров с закрытыми глазами
closed_eye_frames = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    sleeping_count = 0

    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, frame.shape)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, frame.shape)
            avg_ear = (left_ear + right_ear) / 2.0

            # Инициализация словаря
            if idx not in closed_eye_frames:
                closed_eye_frames[idx] = 0

            # Проверка, закрыт ли глаз
            if avg_ear < EAR_THRESHOLD:
                closed_eye_frames[idx] += 1
            else:
                closed_eye_frames[idx] = 0

            # Если глаза закрыты долго → спящий
            if closed_eye_frames[idx] >= CLOSED_FRAMES:
                sleeping_count += 1
                cv2.putText(frame, "СПИТ", (50, 50 + idx*20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            # Рисуем точки лица
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Обновляем Tkinter
    label.config(text=f"Спят: {sleeping_count}/{MAX_FACES}")
    root.update()

    # Показываем видео
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
"""
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

# ----------------------
# Настройки
# ----------------------
EAR_THRESHOLD = 0.25       # порог закрытого глаза
CLOSED_FRAMES = 30         # количество кадров для "спящего" (примерно 3 сек при 10 fps)
MAX_FACES = 20

# ----------------------
# Инициализация Face Mesh
# ----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------
# Индексы глаз (MediaPipe 468 landmarks)
# ----------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ----------------------
# Tkinter окно для отображения спящих учеников
# ----------------------
root = tk.Tk()
root.title("Спящие ученики")
label = tk.Label(root, text="Спят: 0/20", font=("Arial", 24))
label.pack()

# ----------------------
# Камера
# ----------------------
cap = cv2.VideoCapture(0)

# Словарь для хранения количества кадров с закрытыми глазами
closed_eye_frames = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    sleeping_count = 0
    sleeping_faces = []

    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # EAR
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, frame.shape)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, frame.shape)
            avg_ear = (left_ear + right_ear) / 2.0

            # Инициализация словаря
            if idx not in closed_eye_frames:
                closed_eye_frames[idx] = 0

            if avg_ear < EAR_THRESHOLD:
                closed_eye_frames[idx] += 1
            else:
                closed_eye_frames[idx] = 0

            # Если глаза закрыты долго → спящий
            if closed_eye_frames[idx] >= CLOSED_FRAMES:
                sleeping_count += 1

                # Вырезаем лицо и добавляем в список спящих
                h, w, _ = frame.shape
                x_vals = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_vals = [int(lm.y * h) for lm in face_landmarks.landmark]
                x_min, x_max = max(min(x_vals)-20,0), min(max(x_vals)+20,w)
                y_min, y_max = max(min(y_vals)-20,0), min(max(y_vals)+20,h)
                face_crop = frame[y_min:y_max, x_min:x_max]
                sleeping_faces.append(face_crop)

            # Рисуем точки лица
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # ----------------------
    # Показываем спящих в отдельном окне
    # ----------------------
    # ----------------------
    # Показываем спящих в отдельном окне
    # ----------------------
    if sleeping_faces:
        # собираем всех спящих в один кадр
        combined = np.zeros((200, 200*len(sleeping_faces), 3), dtype=np.uint8)
        for i, face_crop in enumerate(sleeping_faces):
            face_crop_resized = cv2.resize(face_crop, (200,200))
            combined[:, i*200:(i+1)*200] = face_crop_resized
        cv2.imshow("Спящие ученики", combined)
    else:
        # если спящих нет — просто показываем пустое окно
        empty_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imshow("Спящие ученики", empty_frame)


    # ----------------------
    # Обновляем Tkinter
    # ----------------------
    label.config(text=f"Спят: {sleeping_count}/{MAX_FACES}")
    root.update()

    # ----------------------
    # Показываем видео с точками
    # ----------------------
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()


