import cv2

from src.camera import open_camera, get_frame
from src.hand_tracking import get_hand_landmarks, draw_hand_landmarks
from src.controller import determinate_gest_point_finger, click_l_r, activity
from src.controller_for_model import normalize_coords, determinate

cap = open_camera()

while True:
    frame = get_frame(cap)

    if frame is None:
        break

    hands_data = get_hand_landmarks(frame)

    draw_hand_landmarks(frame, hands_data)

    left_hand = hands_data.get("Left")
    right_hand = hands_data.get("Right")

    if right_hand:
        determinate_gest_point_finger(right_hand)
        click_l_r(right_hand)

    if left_hand:
        norm_coords = normalize_coords(left_hand)
        class_gest = determinate(norm_coords)

        activity(class_gest)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()