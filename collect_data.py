
import cv2
import mediapipe as mp
import csv
import os
import math

# Настройки
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
CAMERA_ID = 0

CLASSES = {
    ord("1"): "fist",
    ord("2"): "open_palm",
    ord("3"): "peace"
}

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_coords(coords):
    """Нормализует 63 координаты относительно запястья."""

    if len(coords) != 63:
        return None

    if not all(math.isfinite(v) for v in coords):
        return None

    x0, y0, z0 = coords[0:3]
    x9, y9, z9 = coords[27:30]

    s = math.sqrt(
        (x9 - x0) ** 2 +
        (y9 - y0) ** 2 +
        (z9 - z0) ** 2
    )

    if s < 1e-6:
        return None

    normalized = []

    for i in range(0, 63, 3):
        x = (coords[i] - x0) / s
        y = (coords[i + 1] - y0) / s
        z = (coords[i + 2] - z0) / s

        normalized.extend([x, y, z])

    return normalized


def get_left_landmarks(results):
    """Извлекает координаты физической левой руки."""

    if not results.multi_hand_landmarks:
        return None

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        hand_label = handedness.classification[0].label

        if hand_label == "Left":
            coords = []

            for landmark in hand_landmarks.landmark:
                coords.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            return coords

    return None


def create_dataset():
    """Создаёт CSV с заголовками, если файла ещё нет."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        headers = [
            f"{axis}{i}"
            for i in range(21)
            for axis in ("x", "y", "z")
        ]
        headers.append("label")

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def get_counts():
    """Подсчитывает уже сохранённые примеры."""

    counts = {label: 0 for label in CLASSES.values()}

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row.get("label")
            if label in counts:
                counts[label] += 1

    return counts


def save_sample(coords, label):
    """Сохраняет один пример в CSV."""

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(coords + [label])


def main():
    create_dataset()
    counts = get_counts()

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        return

    selected_class = None
    current_coords = None

    print("Сбор датасета запущен.")
    print("1 - fist | 2 - open_palm | 3 - peace")
    print("S - сохранить пример | Q - выход")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Ошибка получения кадра.")
                break

            # Зеркальное отображение для удобства
            frame = cv2.flip(frame, 1)

            # MediaPipe ожидает RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_coords = None

            # Рисуем найденные руки
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                # Получаем координаты левой руки
                raw_coords = get_left_landmarks(results)

                if raw_coords is not None:
                    current_coords = normalize_coords(raw_coords)

            # Информация на экране
            status = (
                f"Class: {selected_class}"
                if selected_class
                else "Select class: 1 / 2 / 3"
            )

            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            hand_status = (
                "Left hand: detected"
                if current_coords is not None
                else "Left hand: not detected"
            )

            cv2.putText(
                frame, hand_status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )

            # Счётчики
            y_pos = 105
            for label, count in counts.items():
                cv2.putText(
                    frame,
                    f"{label}: {count}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1
                )
                y_pos += 30

            cv2.putText(
                frame, "S - save | Q - quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Выбор класса
            if key in CLASSES:
                selected_class = CLASSES[key]
                print(f"Выбран класс: {selected_class}")

            # Сохранение примера
            elif key == ord("s"):
                if selected_class is None:
                    print("Сначала выбери класс (1, 2 или 3).")

                elif current_coords is None:
                    print("Левая рука не обнаружена или координаты некорректны.")

                elif len(current_coords) != 63:
                    print("Ошибка: неверное количество признаков.")

                else:
                    save_sample(current_coords, selected_class)
                    counts[selected_class] += 1

                    print(
                        f"Сохранён {selected_class}: "
                        f"{counts[selected_class]} примеров"
                    )

            # Выход
            elif key == ord("q"):
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

        print("\nСбор данных завершён.")
        print("Итоговое количество примеров:")

        for label, count in counts.items():
            print(f"{label}: {count}")

        print(f"Датасет сохранён в: {CSV_PATH}")


if __name__ == "__main__":
    main()