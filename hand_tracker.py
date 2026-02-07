import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time


class FingerMouseController:
    def __init__(self, cam_width=960, cam_height=720, smoothing_window=7):

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # Screen
        self.screen_width, self.screen_height = pyautogui.size()

        # Camera
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.margin = 100

        # Smoothing
        self.x_history = deque(maxlen=smoothing_window)
        self.y_history = deque(maxlen=smoothing_window)

        # Gesture
        self.click_frames_required = 4
        self.left_touch_frames = 0
        self.right_touch_frames = 0
        self.click_cooldown = 0.5
        self.last_click_time = 0

        # Landmarks
        self.INDEX_TIP = 8
        self.THUMB_TIP = 4
        self.MIDDLE_TIP = 12
        self.WRIST = 0
        self.MIDDLE_MCP = 9

        # FPS
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    # ---------- Utility ----------

    def reset_smoothing(self):
        self.x_history.clear()
        self.y_history.clear()

    def smooth(self, x, y):
        self.x_history.append(x)
        self.y_history.append(y)
        sx = sum(self.x_history) / len(self.x_history)
        sy = sum(self.y_history) / len(self.y_history)
        return int(sx), int(sy)

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

    def can_click(self):
        return (time.time() - self.last_click_time) > self.click_cooldown

    def register_click(self):
        self.last_click_time = time.time()

    # ---------- Gesture Math ----------

    def hand_scale(self, lm):
        w = lm.landmark[self.WRIST]
        m = lm.landmark[self.MIDDLE_MCP]
        return np.hypot(w.x - m.x, w.y - m.y)

    def finger_dist(self, lm, a, b):
        p1 = lm.landmark[a]
        p2 = lm.landmark[b]
        return np.hypot(p1.x - p2.x, p1.y - p2.y)

    # ---------- Frame Processing ----------

    def process_frame(self, img):

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        cv2.rectangle(
            img,
            (self.margin, self.margin),
            (self.cam_width - self.margin, self.cam_height - self.margin),
            (255, 0, 255),
            2
        )

        hand_detected = False

        if results.multi_hand_landmarks:

            hand_detected = True
            lm = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                img, lm, self.mp_hands.HAND_CONNECTIONS
            )

            # Cursor mapping
            tip = lm.landmark[self.INDEX_TIP]
            cam_x = int(tip.x * self.cam_width)
            cam_y = int(tip.y * self.cam_height)

            sx = np.interp(
                cam_x,
                (self.margin, self.cam_width - self.margin),
                (0, self.screen_width)
            )

            sy = np.interp(
                cam_y,
                (self.margin, self.cam_height - self.margin),
                (0, self.screen_height)
            )

            sx, sy = self.smooth(sx, sy)
            pyautogui.moveTo(sx, sy)

            # Gesture detection
            scale = self.hand_scale(lm)
            threshold = scale * 0.45

            d_ti = self.finger_dist(lm, self.THUMB_TIP, self.INDEX_TIP)
            d_tm = self.finger_dist(lm, self.THUMB_TIP, self.MIDDLE_TIP)

            # Left click
            if d_ti < threshold:
                self.left_touch_frames += 1
                if self.left_touch_frames >= self.click_frames_required and self.can_click():
                    pyautogui.click()
                    self.register_click()
            else:
                self.left_touch_frames = 0

            # Right click
            if d_tm < threshold:
                self.right_touch_frames += 1
                if self.right_touch_frames >= self.click_frames_required and self.can_click():
                    pyautogui.rightClick()
                    self.register_click()
            else:
                self.right_touch_frames = 0

        else:
            self.reset_smoothing()
            self.left_touch_frames = 0
            self.right_touch_frames = 0

        self.update_fps()

        status = "Hand" if hand_detected else "No Hand"
        cv2.putText(
            img,
            f"FPS:{int(self.fps)} | {status}",
            (20, self.cam_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return img

    # ---------- Main Loop ----------

    def run(self):

        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_width)
        cap.set(4, self.cam_height)

        print("Finger Mouse PRO running — press Q to quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = self.process_frame(frame)
            cv2.imshow("Finger Mouse PRO", out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    FingerMouseController().run()
