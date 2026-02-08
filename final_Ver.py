import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time


class FingerMouseController:

    def __init__(self, cam_width=960, cam_height=720):

        # -------- CONFIG --------
        self.margin = 100
        self.click_frames_required = 2
        self.click_cooldown = 0.35
        self.scroll_gain = 12
        self.drag_enter_frames = 10
        self.pinch_ratio = 0.22

        # -------- SAFETY --------
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        # -------- MEDIAPIPE --------
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.7
        )

        # -------- SYSTEM --------
        self.sw, self.sh = pyautogui.size()
        self.cw, self.ch = cam_width, cam_height

        # -------- SMOOTHING (Adaptive EMA) --------
        self.ema_alpha = 0.35
        self.sx = None
        self.sy = None
        self.prev_rx = None
        self.prev_ry = None
        self.y_hist = deque(maxlen=6)

        # -------- LANDMARK IDS --------
        self.INDEX = 8
        self.THUMB = 4
        self.MIDDLE = 12
        self.WRIST = 0
        self.MCP = 9

        # -------- STATE --------
        self.pinch_frames = 0
        self.right_frames = 0
        self.dragging = False
        self.left_click_fired = False
        self.right_click_fired = False
        self.last_click_time = 0

        # -------- FPS --------
        self.frames = 0
        self.fps = 0
        self.last_fps_time = time.time()

    # ================= UTIL =================

    def smooth(self, x, y):
        self.y_hist.append(y)

        if self.sx is None:
            self.sx, self.sy = x, y
            self.prev_rx, self.prev_ry = x, y
            return int(x), int(y)

        dx = abs(x - self.prev_rx)
        dy = abs(y - self.prev_ry)
        speed = np.hypot(dx, dy)
        self.prev_rx, self.prev_ry = x, y

        alpha = np.clip(self.ema_alpha + speed / 1500.0, self.ema_alpha, 0.85)

        self.sx += alpha * (x - self.sx)
        self.sy += alpha * (y - self.sy)
        return int(self.sx), int(self.sy)

    def dist(self, lm, a, b):
        p1, p2 = lm.landmark[a], lm.landmark[b]
        return np.hypot(p1.x - p2.x, p1.y - p2.y)

    def hand_scale(self, lm):
        return self.dist(lm, self.WRIST, self.MCP)

    def finger_up(self, lm, tip, pip):
        return lm.landmark[tip].y < lm.landmark[pip].y

    def can_click(self):
        return (time.time() - self.last_click_time) > self.click_cooldown

    def mark_click(self):
        self.last_click_time = time.time()

    def release_drag(self):
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False

    def update_fps(self):
        self.frames += 1
        now = time.time()
        if now - self.last_fps_time >= 1:
            self.fps = self.frames / (now - self.last_fps_time)
            self.frames = 0
            self.last_fps_time = now

    # ================= MAIN =================

    def process(self, img):

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        gesture = "MOVE"

        cv2.rectangle(img,(self.margin,self.margin),
                      (self.cw-self.margin,self.ch-self.margin),
                      (255,0,255),2)

        if res.multi_hand_landmarks:

            lm = res.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(img,lm,self.mp_hands.HAND_CONNECTIONS)

            tip = lm.landmark[self.INDEX]
            cx, cy = int(tip.x*self.cw), int(tip.y*self.ch)

            if not (self.margin < cx < self.cw-self.margin and
                    self.margin < cy < self.ch-self.margin):
                self.release_drag()
                return img

            sx = np.interp(cx,(self.margin,self.cw-self.margin),(0,self.sw))
            sy = np.interp(cy,(self.margin,self.ch-self.margin),(0,self.sh))
            sx, sy = self.smooth(sx, sy)

            # -------- gesture metrics --------
            scale = self.hand_scale(lm)
            th = scale * self.pinch_ratio

            d_it = self.dist(lm,self.THUMB,self.INDEX)
            d_mt = self.dist(lm,self.THUMB,self.MIDDLE)

            index_up  = self.finger_up(lm,8,6)
            middle_up = self.finger_up(lm,12,10)

            index_pinch  = d_it < th
            middle_pinch = d_mt < th

            # =================================
            # SCROLL — two fingers up
            # =================================
            if index_up and middle_up and not index_pinch and not middle_pinch:
                gesture = "SCROLL"
                self.release_drag()
                self.pinch_frames = 0
                self.right_frames = 0
                self.left_click_fired = False
                self.right_click_fired = False

                if len(self.y_hist) >= 4:
                    dy = np.diff(list(self.y_hist)[-4:])
                    velocity = np.mean(dy)
                    scroll = int(np.clip(velocity * self.scroll_gain, -60, 60))
                    pyautogui.scroll(-scroll)

                pyautogui.moveTo(sx,sy)

            # =================================
            # RIGHT CLICK
            # =================================
            elif middle_pinch and not index_pinch and index_up and not self.dragging:
                gesture = "RIGHT"
                self.pinch_frames = 0
                self.left_click_fired = False

                self.right_frames += 1
                if (self.right_frames >= self.click_frames_required
                        and not self.right_click_fired
                        and self.can_click()):
                    pyautogui.rightClick()
                    self.mark_click()
                    self.right_click_fired = True
                    gesture = "RIGHT CLICK ✓"

                pyautogui.moveTo(sx,sy)

            # =================================
            # LEFT CLICK / DRAG
            # =================================
            elif index_pinch and not middle_pinch:

                self.right_frames = 0
                self.right_click_fired = False
                self.pinch_frames += 1

                if self.pinch_frames >= self.drag_enter_frames:
                    gesture = "DRAG"
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                    pyautogui.moveTo(sx,sy)

                elif self.pinch_frames >= self.click_frames_required:
                    if not self.left_click_fired and self.can_click():
                        pyautogui.click()
                        self.mark_click()
                        self.left_click_fired = True
                        gesture = "LEFT CLICK ✓"
                    pyautogui.moveTo(sx,sy)

                else:
                    gesture = "PINCH"
                    pyautogui.moveTo(sx,sy)

            # =================================
            # MOVE
            # =================================
            else:
                gesture = "MOVE"
                self.release_drag()
                self.pinch_frames = 0
                self.right_frames = 0
                self.left_click_fired = False
                self.right_click_fired = False
                pyautogui.moveTo(sx,sy)

            # -------- debug --------
            cv2.putText(img,f"I-T:{d_it:.3f} M-T:{d_mt:.3f} TH:{th:.3f}",
                        (20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        else:
            self.release_drag()
            self.pinch_frames = 0
            self.right_frames = 0
            self.left_click_fired = False
            self.right_click_fired = False
            self.sx = self.sy = None
            self.prev_rx = self.prev_ry = None
            self.y_hist.clear()

        self.update_fps()
        cv2.putText(img,f"{gesture} | FPS:{int(self.fps)}",
                    (20,self.ch-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        return img

    def run(self):

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not accessible")
            return

        cap.set(3,self.cw)
        cap.set(4,self.ch)

        print("Gesture Mouse Running — press Q to quit")

        try:
            while True:
                ok,frame = cap.read()
                if not ok:
                    break

                out = self.process(frame)
                cv2.imshow("Gesture Mouse", out)

                if cv2.waitKey(1)&0xFF==ord('q'):
                    break

        except pyautogui.FailSafeException:
            print("Failsafe triggered")

        finally:
            self.release_drag()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    FingerMouseController().run()
