import pyautogui

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

def click_l_r(hand_landmarks):
    thumb = hand_landmarks.landmark[4]

    point = hand_landmarks.landmark[8]
    middle = hand_landmarks.landmark[12]

    x_thumb = thumb.x
    y_thumb = thumb.y

    x_point = point.x
    y_point = point.y

    x_middle = middle.x
    y_middle = middle.y

    if ((x_thumb-x_point)**2 + (y_thumb-y_point)**2)**0.5 < 0.05:
        pyautogui.click()
    if ((x_thumb-x_middle)**2 + (y_thumb-y_middle)**2)**0.5 < 0.05:
        pyautogui.rightClick()