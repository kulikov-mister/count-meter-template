# pip install opencv-python numpy dlib

import cv2
import numpy as np
import dlib

# Подключаем камеру (0 - стандартная веб-камера)
cap = cv2.VideoCapture(0)

# Загружаем детектор лиц из библиотеки dlib
detector = dlib.get_frontal_face_detector()

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Отражаем изображение по горизонтали

    # Переводим в оттенки серого (ускоряет обработку)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = detector(gray)

    # Счётчик лиц
    face_count = len(faces)

    for i, face in enumerate(faces, start=1):
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0),
                      2)  # Рисуем рамку вокруг лица
        cv2.putText(frame, f"Лицо {i}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Подписываем лица

    # Выводим количество лиц на экране
    cv2.putText(frame, f"Лиц: {face_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Показываем кадр с обнаруженными лицами
    cv2.imshow("Обнаружение лиц", frame)

    # Нажатие 'q' завершает программу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
