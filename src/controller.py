import pyautogui
import time

x_old = None
y_old = None

def determinate_gest_point_finger(hand_landmarks):

    global x_old, y_old

    finger = hand_landmarks.landmark[8]

    x = finger.x
    y = finger.y
    
    if x_old is None or y_old is None:
        x_old = x
        y_old = y
        return

    delta_x = x - x_old
    delta_y = y - y_old

    sensitivity = 4000

    pyautogui.moveRel(
        delta_x * sensitivity,
        delta_y * sensitivity
    )

    x_old = x
    y_old = y

pinkie_touch_old = False
ring_touch_old = False


def click_l_r(hand_landmarks):
    global pinkie_touch_old, ring_touch_old

    thumb = hand_landmarks.landmark[4]
    ring = hand_landmarks.landmark[16]
    pinkie = hand_landmarks.landmark[20]

    pinkie_dist = (
        (thumb.x - pinkie.x) ** 2 +
        (thumb.y - pinkie.y) ** 2
    ) ** 0.5

    ring_dist = (
        (thumb.x - ring.x) ** 2 +
        (thumb.y - ring.y) ** 2
    ) ** 0.5

    pinkie_touch = pinkie_dist < 0.05
    ring_touch = ring_dist < 0.05

    if pinkie_touch and not pinkie_touch_old:
        pyautogui.click()

    if ring_touch and not ring_touch_old:
        pyautogui.rightClick()

    pinkie_touch_old = pinkie_touch
    ring_touch_old = ring_touch

last_action_time = 0
previous_gesture = None
cooldown = 1.0


def activity(gesture):
    global last_action_time, previous_gesture

    current_time = time.time()

    if gesture is None:
        previous_gesture = None
        return

    if gesture == "fist":
        if gesture != previous_gesture:
            if current_time - last_action_time >= cooldown:
                pyautogui.hotkey("alt", "tab")
                last_action_time = current_time

        previous_gesture = gesture
        return

    elif gesture == "open_palm":
        if current_time - last_action_time >= 0.2:
            pyautogui.scroll(40)
            last_action_time = current_time

    elif gesture == "peace":
        if current_time - last_action_time >= 0.2:
            pyautogui.scroll(-40)
            last_action_time = current_time

    previous_gesture = gesture