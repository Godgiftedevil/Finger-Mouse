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
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        # Screen & Camera
        self.sw, self.sh = pyautogui.size()
        self.cw, self.ch = cam_width, cam_height
        self.margin = 100

        # Smoothing
        self.x_hist = deque(maxlen=smoothing_window)
        self.y_hist = deque(maxlen=smoothing_window)

        # Landmarks
        self.INDEX = 8
        self.THUMB = 4
        self.MIDDLE = 12
        self.WRIST = 0
        self.MCP = 9

        # Gesture State
        self.click_frames_required = 4
        self.left_frames = 0
        self.right_frames = 0
        self.drag_frames = 0
        self.dragging = False
        self.scroll_mode = False  # Track scroll state separately

        self.click_cooldown = 0.5
        self.last_click_time = 0

        # Performance
        self.frames = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def smooth(self, x, y):
        self.x_hist.append(x)
        self.y_hist.append(y)
        return (int(np.mean(self.x_hist)), int(np.mean(self.y_hist)))

    def reset_gesture_state(self):
        """Reset only gesture counters, not smoothing buffers"""
        self.left_frames = self.right_frames = self.drag_frames = 0

    def reset_smoothing(self):
        self.x_hist.clear()
        self.y_hist.clear()

    def can_click(self):
        return (time.time() - self.last_click_time) > self.click_cooldown

    def mark_click(self):
        self.last_click_time = time.time()

    def finger_distance(self, lm, id1, id2):
        p1, p2 = lm.landmark[id1], lm.landmark[id2]
        return np.hypot(p1.x - p2.x, p1.y - p2.y)

    def hand_scale(self, lm):
        return self.finger_distance(lm, self.WRIST, self.MCP)

    def update_fps(self):
        self.frames += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frames / (now - self.last_fps_time)
            self.frames = 0
            self.last_fps_time = now

    def process(self, img):
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture_text = "MOVE"
        hand_detected = False

        # Draw control area
        cv2.rectangle(img, (self.margin, self.margin),
                     (self.cw - self.margin, self.ch - self.margin),
                     (255, 0, 255), 2)

        if results.multi_hand_landmarks:
            hand_detected = True
            lm = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(img, lm, self.mp_hands.HAND_CONNECTIONS)

            # Cursor mapping
            tip = lm.landmark[self.INDEX]
            cam_x = int(tip.x * self.cw)
            cam_y = int(tip.y * self.ch)

            screen_x = np.interp(cam_x, (self.margin, self.cw - self.margin), (0, self.sw))
            screen_y = np.interp(cam_y, (self.margin, self.ch - self.margin), (0, self.sh))
            screen_x = np.clip(screen_x, 0, self.sw - 1)
            screen_y = np.clip(screen_y, 0, self.sh - 1)

            smooth_x, smooth_y = self.smooth(screen_x, screen_y)
            
            # Only move cursor if not dragging (to avoid jumpiness)
            if not self.dragging:
                pyautogui.moveTo(smooth_x, smooth_y)

            # Gesture detection
            scale = self.hand_scale(lm)
            thumb_index_dist = self.finger_distance(lm, self.THUMB, self.INDEX)
            thumb_middle_dist = self.finger_distance(lm, self.THUMB, self.MIDDLE)
            threshold = scale * 0.45

            # =====================
            # SCROLL MODE (Two fingers extended)
            # =====================
            if thumb_index_dist > threshold and thumb_middle_dist > threshold * 1.1:
                if not self.scroll_mode:
                    self.reset_gesture_state()
                    self.scroll_mode = True
                
                gesture_text = "SCROLL"
                if len(self.y_hist) > 1:
                    scroll_amount = int((self.y_hist[-1] - self.y_hist[-2]) * 10)  # More responsive
                    pyautogui.scroll(-scroll_amount)

            # =====================
            # DRAG MODE (Pinch thumb-index)
            # =====================
            elif thumb_index_dist < threshold:
                self.drag_frames += 1
                self.scroll_mode = False
                
                if self.drag_frames >= self.click_frames_required:
                    gesture_text = "DRAG"
                    if not self.dragging:
                        pyautogui.mouseDown()
                        self.dragging = True
                pyautogui.moveTo(smooth_x, smooth_y)  # Move during drag

            else:
                # Release drag if was dragging
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                self.drag_frames = 0
                self.scroll_mode = False

            # =====================
            # RIGHT CLICK (Pinch thumb-middle)
            # =====================
            if thumb_middle_dist < threshold and thumb_index_dist > threshold:
                self.right_frames += 1
                if self.right_frames >= self.click_frames_required and self.can_click():
                    pyautogui.rightClick()
                    self.mark_click()
                    gesture_text = "RIGHT CLICK"
                    self.reset_gesture_state()
            else:
                self.right_frames = 0

            # =====================
            # LEFT CLICK (Only if not dragging)
            # =====================
            if thumb_index_dist < threshold and not self.dragging:
                self.left_frames += 1
                if self.left_frames >= self.click_frames_required and self.can_click():
                    pyautogui.click()
                    self.mark_click()
                    gesture_text = "LEFT CLICK"
            else:
                self.left_frames = 0

        else:
            # No hand detected - reset all states
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            self.reset_gesture_state()
            self.reset_smoothing()
            self.scroll_mode = False
            hand_detected = False

        # Display
        self.update_fps()
        status = f"{gesture_text} | FPS: {int(self.fps)}"
        if not hand_detected:
            status = "NO HAND | " + status
            
        cv2.putText(img, status, (20, self.ch - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ch)

        print("Phase 3 — Drag + Scroll Enabled")
        print("Controls: Two fingers = Scroll | Pinch thumb-index = Drag | Pinch thumb-middle = Right click")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                out_frame = self.process(frame)
                cv2.imshow("Finger Mouse Phase 3", out_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except pyautogui.FailSafeException:
            pass
        finally:
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    FingerMouseController().run()
