import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time


class FingerMouseController:

    def __init__(self, cam_width=960, cam_height=720, smoothing_window=7):

        # -------- CONFIG --------
        self.margin = 100
        self.click_frames_required = 4
        self.click_cooldown = 0.5
        self.scroll_gain = 8
        self.drag_threshold_ratio = 0.45

        # -------- SAFETY --------
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        # -------- MEDIAPIPE --------
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # -------- SYSTEM --------
        self.sw, self.sh = pyautogui.size()
        self.cw, self.ch = cam_width, cam_height

        # -------- SMOOTHING --------
        self.x_hist = deque(maxlen=smoothing_window)
        self.y_hist = deque(maxlen=smoothing_window)

        # -------- LANDMARK IDS --------
        self.INDEX = 8
        self.THUMB = 4
        self.MIDDLE = 12
        self.WRIST = 0
        self.MCP = 9

        # -------- STATE --------
        self.left_frames = 0
        self.right_frames = 0
        self.drag_frames = 0
        self.dragging = False
        self.scroll_mode = False
        self.last_click_time = 0

        # -------- HAND DEBOUNCE --------
        self.no_hand_frames = 0
        self.no_hand_reset_frames = 3

        # -------- FPS --------
        self.frames = 0
        self.fps = 0
        self.last_fps_time = time.time()

    # ================= UTIL =================

    def smooth(self, x, y):
        self.x_hist.append(x)
        self.y_hist.append(y)
        return int(np.mean(self.x_hist)), int(np.mean(self.y_hist))

    def reset_states(self):
        self.left_frames = self.right_frames = self.drag_frames = 0

    def reset_smoothing(self):
        self.x_hist.clear()
        self.y_hist.clear()

    def can_click(self):
        return (time.time() - self.last_click_time) > self.click_cooldown

    def mark_click(self):
        self.last_click_time = time.time()

    def dist(self, lm, a, b):
        p1, p2 = lm.landmark[a], lm.landmark[b]
        return np.hypot(p1.x-p2.x, p1.y-p2.y)

    def hand_scale(self, lm):
        return self.dist(lm, self.WRIST, self.MCP)

    def update_fps(self):
        self.frames += 1
        now = time.time()
        if now - self.last_fps_time >= 1:
            self.fps = self.frames/(now-self.last_fps_time)
            self.frames = 0
            self.last_fps_time = now

    # ================= MAIN =================

    def process(self, img):

        img = cv2.flip(img,1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        gesture = "MOVE"

        cv2.rectangle(img,(self.margin,self.margin),
                      (self.cw-self.margin,self.ch-self.margin),
                      (255,0,255),2)

        if res.multi_hand_landmarks:

            self.no_hand_frames = 0
            lm = res.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(img,lm,self.mp_hands.HAND_CONNECTIONS)

            tip = lm.landmark[self.INDEX]
            cx = int(tip.x*self.cw)
            cy = int(tip.y*self.ch)

            if not (self.margin < cx < self.cw-self.margin and
                    self.margin < cy < self.ch-self.margin):
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                self.reset_states()
                self.scroll_mode = False
                return img

            sx = np.interp(cx,(self.margin,self.cw-self.margin),(0,self.sw))
            sy = np.interp(cy,(self.margin,self.ch-self.margin),(0,self.sh))
            sx = int(np.clip(sx, 0, self.sw - 1))
            sy = int(np.clip(sy, 0, self.sh - 1))
            sx,sy = self.smooth(sx,sy)

            if not self.dragging:
                pyautogui.moveTo(sx,sy)

            scale = self.hand_scale(lm)
            th = scale * self.drag_threshold_ratio
            d_ti = self.dist(lm,self.THUMB,self.INDEX)
            d_tm = self.dist(lm,self.THUMB,self.MIDDLE)

            # ---- SCROLL ----
            if d_ti > th and d_tm > th*1.1:
                if not self.scroll_mode:
                    self.reset_states()
                    self.scroll_mode = True
                gesture = "SCROLL"
                if len(self.y_hist)>1:
                    delta = (self.y_hist[-1]-self.y_hist[-2])*self.scroll_gain
                    delta = int(np.clip(delta,-40,40))
                    pyautogui.scroll(-delta)

            # ---- DRAG ----
            elif d_ti < th:
                self.scroll_mode = False
                self.drag_frames += 1
                if self.drag_frames >= self.click_frames_required:
                    gesture = "DRAG"
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                pyautogui.moveTo(sx,sy)

            else:
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                self.drag_frames = 0
                self.scroll_mode = False

            # ---- RIGHT CLICK ----
            if not self.dragging and not self.scroll_mode:
                if d_tm < th and d_ti > th:
                    self.right_frames += 1
                    if self.right_frames>=self.click_frames_required and self.can_click():
                        pyautogui.rightClick()
                        self.mark_click()
                        gesture="RIGHT CLICK"
                else:
                    self.right_frames=0

            # ---- LEFT CLICK ----
            if not self.dragging and not self.scroll_mode:
                if d_ti < th and self.drag_frames < self.click_frames_required:
                    self.left_frames += 1
                    if self.left_frames>=self.click_frames_required and self.can_click():
                        pyautogui.click()
                        self.mark_click()
                        gesture="LEFT CLICK"
                else:
                    self.left_frames=0

        else:
            self.no_hand_frames += 1
            if self.no_hand_frames >= self.no_hand_reset_frames:
                if self.dragging:
                    pyautogui.mouseUp()
                self.dragging = False
                self.reset_states()
                self.scroll_mode = False
                self.reset_smoothing()

        self.update_fps()
        cv2.putText(img,f"{gesture} | FPS:{int(self.fps)}",
                    (20,self.ch-20),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)

        return img

    def run(self):

        cap = cv2.VideoCapture(0)
        cap.set(3,self.cw)
        cap.set(4,self.ch)

        print("FINAL PRODUCTION BUILD RUNNING")

        try:
            while True:
                ok,frame = cap.read()
                if not ok:
                    break

                out=self.process(frame)
                cv2.imshow("Finger Mouse Final",out)

                if cv2.waitKey(1)&0xFF==ord('q'):
                    break

        except pyautogui.FailSafeException:
            print("Failsafe triggered")

        finally:
            if self.dragging:
                pyautogui.mouseUp()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    FingerMouseController().run()
