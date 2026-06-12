#!/usr/bin/env python3

import subprocess
import time
import tkinter as tk

import cv2
import mediapipe as mp
from picamera2 import Picamera2


# Gesture tuning
#
# Hysteresis:
# - PINCH_ON_THRESHOLD: distance at or below this means "pinched"
# - PINCH_RELEASE_THRESHOLD: distance at or above this means "released"
# - Between the two values, keep the previous state to avoid jitter.
PINCH_ON_THRESHOLD = 0.09
PINCH_RELEASE_THRESHOLD = 0.18
PINCH_HOLD_SECONDS = 0.15
KEYPRESS_COOLDOWN_SECONDS = 0.75

# Keypress target
KEY_TO_SEND = "f"
TARGET_WINDOW_NAME = "asciiquarium-target"

# Debug output
DEBUG = False
DEBUG_INTERVAL_SECONDS = 0.5

# Simple status overlay
# Adjust X and Y until the flag lands on the castle flag position.
STATUS_WINDOW_X = 510
STATUS_WINDOW_Y = 430
STATUS_WINDOW_WIDTH = 40
STATUS_WINDOW_HEIGHT = 40

NO_HAND_FLAG = "⚐"
HAND_FLAG = "⚑"


def send_keypress() -> None:
    """Send the configured key to the target xterm window using xdotool."""
    print("Sending F key")

    subprocess.run(
        [
            "xdotool",
            "search",
            "--name",
            TARGET_WINDOW_NAME,
            "key",
            "--window",
            "%1",
            KEY_TO_SEND,
        ],
        check=False,
    )


def create_status_window():
    """Create a tiny always-on-top overlay window."""
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.geometry(
        f"{STATUS_WINDOW_WIDTH}x{STATUS_WINDOW_HEIGHT}"
        f"+{STATUS_WINDOW_X}+{STATUS_WINDOW_Y}"
    )

    label = tk.Label(
        root,
        text=NO_HAND_FLAG,
        font=("DejaVu Sans", 24, "bold"),
        fg="lime",
        bg="black",
        bd=0,
        padx=0,
        pady=0,
    )
    label.pack(fill="both", expand=True)

    root.update_idletasks()
    root.update()

    return root, label


def set_status_flag(root, label, hand_detected: bool) -> None:
    """Update the overlay flag."""
    if hand_detected:
        label.config(text=HAND_FLAG)
    else:
        label.config(text=NO_HAND_FLAG)

    root.update_idletasks()
    root.update()


def main() -> None:
    pinch_was_active = False
    pinch_start_time = None
    last_keypress_time = 0.0
    last_debug_time = 0.0

    status_root, status_label = create_status_window()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    )

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration())
    picam2.start()
    time.sleep(0.5)

    print("Pinch/release tracking started.")
    print(f"Target window: {TARGET_WINDOW_NAME}")
    print(f"Key to send: {KEY_TO_SEND}")
    print(f"Pinch-on threshold: {PINCH_ON_THRESHOLD}")
    print(f"Release threshold: {PINCH_RELEASE_THRESHOLD}")
    print(f"Pinch hold time: {PINCH_HOLD_SECONDS}")
    print("Pinch thumb/index finger, then release, to send F.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            frame = picam2.capture_array()

            if frame.ndim != 3:
                status_root.update_idletasks()
                status_root.update()
                continue

            if frame.shape[2] == 4:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb_frame)
            now = time.time()

            if result.multi_hand_landmarks:
                set_status_flag(status_root, status_label, True)

                hand_landmarks = result.multi_hand_landmarks[0]

                index_tip = hand_landmarks.landmark[
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                ]
                thumb_tip = hand_landmarks.landmark[
                    mp.solutions.hands.HandLandmark.THUMB_TIP
                ]

                thumb_dx = thumb_tip.x - index_tip.x
                thumb_dy = thumb_tip.y - index_tip.y
                thumb_dist_sq = thumb_dx * thumb_dx + thumb_dy * thumb_dy
                thumb_dist = thumb_dist_sq ** 0.5

                if thumb_dist <= PINCH_ON_THRESHOLD:
                    gesture_state = "pinching"

                    if pinch_start_time is None:
                        pinch_start_time = now

                    if now - pinch_start_time >= PINCH_HOLD_SECONDS:
                        pinch_was_active = True
                        gesture_state = "pinched/armed"

                elif thumb_dist >= PINCH_RELEASE_THRESHOLD:
                    gesture_state = "released"

                    if pinch_was_active:
                        if now - last_keypress_time >= KEYPRESS_COOLDOWN_SECONDS:
                            send_keypress()
                            last_keypress_time = now

                    pinch_was_active = False
                    pinch_start_time = None

                else:
                    gesture_state = "holding"

                    if not pinch_was_active:
                        pinch_start_time = None

                if DEBUG and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                    hold_time = (
                        0.0 if pinch_start_time is None else now - pinch_start_time
                    )
                    print(
                        f"hand detected: thumb_dist={thumb_dist:.3f}, "
                        f"pinch_on={PINCH_ON_THRESHOLD:.3f}, "
                        f"release={PINCH_RELEASE_THRESHOLD:.3f}, "
                        f"hold={hold_time:.2f}, "
                        f"state={gesture_state}, "
                        f"pinch_was_active={pinch_was_active}"
                    )
                    last_debug_time = now

            else:
                set_status_flag(status_root, status_label, False)

                if DEBUG and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                    print("no hand detected")
                    last_debug_time = now

                pinch_was_active = False
                pinch_start_time = None

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping pinch/release tracking.")

    finally:
        hands.close()
        picam2.stop()
        picam2.close()
        status_root.destroy()


if __name__ == "__main__":
    main()
