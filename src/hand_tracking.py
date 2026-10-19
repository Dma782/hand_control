import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    hands_data = {}

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_label = handedness.classification[0].label

            hands_data[hand_label] = hand_landmarks

    return hands_data

def draw_hand_landmarks(frame, hands_landmarks):
    if hands_landmarks:

        for hand_landmarks in hands_landmarks.values():
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

def get_left_landmarks_for_model(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                if hand_label == 'Left':
                    coords = []
                    for landmark in hand_landmarks.landmark:
                        coords.extend([landmark.x, landmark.y, landmark.z])
                    return coords 
    return []
    
