#!/usr/bin/env python3

import subprocess
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

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

# Shutdown gesture tuning
# Hold a deliberate two-finger V / peace-sign gesture in view:
# - 0-2 seconds: normal lime hand-found flag
# - 2-4 seconds: solid yellow warning flag
# - 4-5 seconds: solid red imminent-shutdown flag
# - 5 seconds: send Q to the target xterm and exit this tracker
SHUTDOWN_YELLOW_SECONDS = 2.0
SHUTDOWN_RED_SECONDS = 4.0
SHUTDOWN_TRIGGER_SECONDS = 5.0
# Allow short MediaPipe dropouts without cancelling the shutdown hold.
SHUTDOWN_GESTURE_GRACE_SECONDS = 0.75
# Shutdown gesture detection tuning.
# Use a deliberate two-finger V / peace-out gesture instead of open palm,
# because an open hand appears naturally while feeding.
# These values are in MediaPipe's normalized hand-coordinate space.
#
# MediaPipe can label angled fingers imperfectly, so do not require the V to
# be specifically index+middle. Treat it as: any two non-thumb fingers are
# clearly extended and spread apart, while not all four fingers are extended.
SHUTDOWN_MIN_EXTENDED_FINGERS = 2
SHUTDOWN_MAX_EXTENDED_FINGERS = 3
SHUTDOWN_EXTENDED_FINGER_RATIO = 1.25
SHUTDOWN_MIN_V_SPREAD = 0.10

# Keypress target
KEY_TO_SEND = "f"
QUIT_KEY_TO_SEND = "q"
TARGET_WINDOW_NAME = "asciiquarium-window"

# Debug output
# DEBUG_MESSAGES prints status/gesture diagnostics to the terminal.
# DEBUG_CAPTURE_FRAMES saves camera frames for gesture debugging.
# Leave DEBUG_CAPTURE_FRAMES off during normal use to avoid filling storage.
DEBUG_MESSAGES = True
DEBUG_CAPTURE_FRAMES = False
DEBUG_INTERVAL_SECONDS = 0.5

# When DEBUG_CAPTURE_FRAMES is True, save one frame after the hand has been
# continuously detected long enough to let you get into the intended shutdown
# gesture position. Shutdown-gesture transition frames are also saved.
DEBUG_HAND_FOUND_SNAPSHOT_SECONDS = 3.0
DEBUG_FRAME_DIR = Path.home() / "openghost_debug_frames"

# Simple status overlay
# Adjust X and Y until the flag lands on the castle flag position.
STATUS_WINDOW_X = 510
STATUS_WINDOW_Y = 430
STATUS_WINDOW_WIDTH = 40
STATUS_WINDOW_HEIGHT = 40

NO_HAND_FLAG = "⚐"
HAND_FLAG = "⚑"

NO_HAND_COLOR = "lime"
HAND_FOUND_COLOR = "lime"
SHUTDOWN_YELLOW_COLOR = "yellow"
SHUTDOWN_RED_COLOR = "red"


HandLandmark = mp.solutions.hands.HandLandmark
DrawingUtils = mp.solutions.drawing_utils


def debug_print(message: str) -> None:
    """Print a debug message only when DEBUG_MESSAGES is enabled."""
    if DEBUG_MESSAGES:
        print(message)




def save_debug_frame(rgb_frame, event_name: str, details: str = "", hand_landmarks=None) -> None:
    """Save a visible RGB camera frame when DEBUG_CAPTURE_FRAMES is enabled."""
    if not DEBUG_CAPTURE_FRAMES:
        return

    DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_event_name = event_name.replace(" ", "_").replace("/", "_")
    filename = DEBUG_FRAME_DIR / f"{timestamp}_{safe_event_name}.png"

    # Save the already-normalized RGB frame used by MediaPipe.
    # Do not preserve a 4th alpha channel from Picamera2, because some
    # viewers display zero-alpha PNGs as blank/transparent images.
    if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 3:
        frame_to_save = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    elif rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
        frame_to_save = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
    else:
        frame_to_save = rgb_frame.copy()

    if hand_landmarks is not None and frame_to_save.ndim == 3:
        DrawingUtils.draw_landmarks(
            frame_to_save,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )

    ok = cv2.imwrite(str(filename), frame_to_save)
    if ok:
        debug_print(f"saved debug frame: {filename} {details}".rstrip())
    else:
        debug_print(f"WARNING: failed to save debug frame: {filename}")

def find_target_windows() -> list[str]:
    """Return xdotool window IDs matching the target xterm window name."""
    result = subprocess.run(
        ["xdotool", "search", "--name", TARGET_WINDOW_NAME],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    window_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not window_ids:
        debug_print(f"WARNING: no xdotool window found for {TARGET_WINDOW_NAME!r}")
    else:
        debug_print(
            f"xdotool found {len(window_ids)} target window(s) for "
            f"{TARGET_WINDOW_NAME!r}: {', '.join(window_ids)}"
        )

    return window_ids


def send_key_to_target(key: str, label: str) -> bool:
    """Send a key to the target xterm window using xdotool."""
    print(f"Sending {label} key")

    sent = False

    for window_id in find_target_windows():
        debug_print(f"Sending {label} to window {window_id}")

        # Activating first is more reliable for terminal applications than
        # relying only on xdotool's --window argument.  Then send both a
        # targeted key event and a focused-window key event.
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", window_id],
            check=False,
            timeout=2,
        )
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "--window", window_id, key],
            check=False,
            timeout=2,
        )
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", key],
            check=False,
            timeout=2,
        )
        sent = True

    return sent


def close_target_windows_after_quit_grace(grace_seconds: float = 1.0) -> None:
    """Close the aquarium xterm if it is still present after q is sent."""
    time.sleep(grace_seconds)

    remaining_windows = find_target_windows()
    if not remaining_windows:
        debug_print("aquarium xterm is gone after quit key")
        return

    print(
        "Aquarium window is still open after Q; "
        "closing the xterm window as a fallback."
    )
    for window_id in remaining_windows:
        subprocess.run(
            ["xdotool", "windowclose", window_id],
            check=False,
            timeout=2,
        )


def send_keypress() -> None:
    """Send the configured feed key to the target xterm window."""
    send_key_to_target(KEY_TO_SEND, KEY_TO_SEND.upper())


def send_quit_keypress() -> None:
    """Send q to the aquarium, then close the xterm if q did not exit it."""
    if send_key_to_target(QUIT_KEY_TO_SEND, QUIT_KEY_TO_SEND.upper()):
        close_target_windows_after_quit_grace()


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
        fg=NO_HAND_COLOR,
        bg="black",
        bd=0,
        padx=0,
        pady=0,
    )
    label.pack(fill="both", expand=True)

    root.update_idletasks()
    root.update()

    return root, label


def set_status_flag(root, label, hand_detected: bool, color: str | None = None) -> None:
    """Update the overlay flag."""
    if hand_detected:
        label.config(text=HAND_FLAG, fg=color or HAND_FOUND_COLOR)
    else:
        label.config(text=NO_HAND_FLAG, fg=color or NO_HAND_COLOR)

    root.update_idletasks()
    root.update()


def landmark_distance(hand_landmarks, first_id, second_id) -> float:
    """Return 2D distance between two MediaPipe hand landmarks."""
    first = hand_landmarks.landmark[first_id]
    second = hand_landmarks.landmark[second_id]
    dx = first.x - second.x
    dy = first.y - second.y
    return (dx * dx + dy * dy) ** 0.5


def finger_extension_ratio(hand_landmarks, mcp_id, pip_id, tip_id) -> float:
    """Return how far the fingertip is from its base relative to the PIP joint.

    A straight finger should put the fingertip much farther from the MCP/base
    than the PIP joint is. A curled finger usually has a lower ratio.
    This works better than comparing y-coordinates when the camera is rotated
    or the hand is held at an angle.
    """
    base_to_pip = landmark_distance(hand_landmarks, mcp_id, pip_id)
    base_to_tip = landmark_distance(hand_landmarks, mcp_id, tip_id)
    if base_to_pip <= 0.0001:
        return 0.0
    return base_to_tip / base_to_pip


def evaluate_shutdown_gesture(hand_landmarks):
    """Return shutdown-gesture decision plus debug measurements.

    Shutdown gesture rules:
    - use a deliberate two-finger V / peace-out gesture;
    - accept any two non-thumb fingers as the V, because MediaPipe can label
      angled fingers imperfectly;
    - require the two extended fingertips to be spread apart;
    - reject a full open palm by requiring that not all four fingers are
      counted as extended.
    """
    ratios = {
        "index": finger_extension_ratio(
            hand_landmarks,
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_TIP,
        ),
        "middle": finger_extension_ratio(
            hand_landmarks,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_TIP,
        ),
        "ring": finger_extension_ratio(
            hand_landmarks,
            HandLandmark.RING_FINGER_MCP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_TIP,
        ),
        "pinky": finger_extension_ratio(
            hand_landmarks,
            HandLandmark.PINKY_MCP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_TIP,
        ),
    }

    tip_ids = {
        "index": HandLandmark.INDEX_FINGER_TIP,
        "middle": HandLandmark.MIDDLE_FINGER_TIP,
        "ring": HandLandmark.RING_FINGER_TIP,
        "pinky": HandLandmark.PINKY_TIP,
    }

    thumb_index_distance = landmark_distance(
        hand_landmarks,
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_FINGER_TIP,
    )

    extended_fingers = [
        name for name, ratio in ratios.items()
        if ratio >= SHUTDOWN_EXTENDED_FINGER_RATIO
    ]
    extended_count = len(extended_fingers)

    best_pair = None
    best_pair_spread = 0.0
    for i, first_name in enumerate(extended_fingers):
        for second_name in extended_fingers[i + 1:]:
            spread = landmark_distance(
                hand_landmarks,
                tip_ids[first_name],
                tip_ids[second_name],
            )
            if spread > best_pair_spread:
                best_pair_spread = spread
                best_pair = (first_name, second_name)

    enough_extended = extended_count >= SHUTDOWN_MIN_EXTENDED_FINGERS
    not_open_palm = extended_count <= SHUTDOWN_MAX_EXTENDED_FINGERS
    spread_ok = best_pair_spread >= SHUTDOWN_MIN_V_SPREAD

    shutdown_gesture = enough_extended and not_open_palm and spread_ok

    details = {
        "ratios": ratios,
        "thumb_index_distance": thumb_index_distance,
        "extended_fingers": extended_fingers,
        "extended_count": extended_count,
        "best_pair": best_pair,
        "best_pair_spread": best_pair_spread,
        "enough_extended": enough_extended,
        "not_open_palm": not_open_palm,
        "spread_ok": spread_ok,
    }
    return shutdown_gesture, details

def format_shutdown_debug(details) -> str:
    """Format shutdown-gesture measurements for concise debug output."""
    ratios = details["ratios"]
    best_pair = details["best_pair"]
    best_pair_text = "none" if best_pair is None else "+".join(best_pair)
    extended_text = ",".join(details["extended_fingers"]) or "none"
    return (
        f"thumb_index={details['thumb_index_distance']:.3f}, "
        f"extended={details['extended_count']}/4 [{extended_text}], "
        f"best_pair={best_pair_text}, "
        f"v_spread={details['best_pair_spread']:.3f}, "
        f"index_ratio={ratios['index']:.2f}, "
        f"middle_ratio={ratios['middle']:.2f}, "
        f"ring_ratio={ratios['ring']:.2f}, "
        f"pinky_ratio={ratios['pinky']:.2f}, "
        f"checks=enough:{details['enough_extended']} "
        f"not_open_palm:{details['not_open_palm']} "
        f"spread:{details['spread_ok']}"
    )

def main() -> None:
    pinch_was_active = False
    pinch_start_time = None
    last_keypress_time = 0.0
    last_debug_time = 0.0

    shutdown_hold_start_time = None
    shutdown_last_seen_time = None
    shutdown_status = "none"
    shutdown_gesture_frame_active = False

    hand_found_start_time = None
    hand_found_snapshot_saved = False

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
    print(f"Feed key to send: {KEY_TO_SEND}")
    print(f"Quit key to send: {QUIT_KEY_TO_SEND}")
    print(f"Pinch-on threshold: {PINCH_ON_THRESHOLD}")
    print(f"Release threshold: {PINCH_RELEASE_THRESHOLD}")
    print(f"Pinch hold time: {PINCH_HOLD_SECONDS}")
    print(f"Shutdown yellow threshold: {SHUTDOWN_YELLOW_SECONDS}")
    print(f"Shutdown red threshold: {SHUTDOWN_RED_SECONDS}")
    print(f"Shutdown trigger threshold: {SHUTDOWN_TRIGGER_SECONDS}")
    print(f"Shutdown gesture grace period: {SHUTDOWN_GESTURE_GRACE_SECONDS}")
    print(f"Shutdown minimum extended fingers: {SHUTDOWN_MIN_EXTENDED_FINGERS}")
    print(f"Shutdown maximum extended fingers: {SHUTDOWN_MAX_EXTENDED_FINGERS}")
    print(f"Shutdown extended-finger ratio: {SHUTDOWN_EXTENDED_FINGER_RATIO}")
    print(f"Shutdown minimum V spread: {SHUTDOWN_MIN_V_SPREAD}")
    if DEBUG_CAPTURE_FRAMES:
        print(f"Debug frames will be saved in: {DEBUG_FRAME_DIR}")
        print(
            "A hand-found debug frame will be saved after "
            f"{DEBUG_HAND_FOUND_SNAPSHOT_SECONDS:.1f}s of continuous hand detection."
        )
    print("Pinch thumb/index finger, then release, to send F.")
    print("Hold a two-finger V / peace-out gesture for 5 seconds to quit the aquarium.")
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
                hand_landmarks = result.multi_hand_landmarks[0]
                shutdown_gesture, gesture_details = evaluate_shutdown_gesture(hand_landmarks)
                gesture_debug = format_shutdown_debug(gesture_details)

                if hand_found_start_time is None:
                    hand_found_start_time = now
                    hand_found_snapshot_saved = False
                    debug_print("hand-found timer started")

                hand_found_time = now - hand_found_start_time
                if (
                    DEBUG_CAPTURE_FRAMES
                    and not hand_found_snapshot_saved
                    and hand_found_time >= DEBUG_HAND_FOUND_SNAPSHOT_SECONDS
                ):
                    save_debug_frame(
                        rgb_frame,
                        "hand_found_after_3s",
                        f"hand_found_time={hand_found_time:.2f}s, {gesture_debug}",
                        hand_landmarks,
                    )
                    hand_found_snapshot_saved = True

                if shutdown_gesture and not shutdown_gesture_frame_active:
                    save_debug_frame(
                        rgb_frame,
                        "shutdown_gesture_detected",
                        gesture_debug,
                        hand_landmarks,
                    )
                    shutdown_gesture_frame_active = True
                elif not shutdown_gesture and shutdown_gesture_frame_active:
                    save_debug_frame(
                        rgb_frame,
                        "shutdown_gesture_lost",
                        gesture_debug,
                        hand_landmarks,
                    )
                    shutdown_gesture_frame_active = False

                if shutdown_gesture:
                    shutdown_last_seen_time = now

                    if shutdown_hold_start_time is None:
                        shutdown_hold_start_time = now
                        shutdown_status = "lime"
                        debug_print(
                            "shutdown gesture started: "
                            f"shutdown gesture detected ({gesture_debug})"
                        )

                    shutdown_hold_time = now - shutdown_hold_start_time

                    if shutdown_hold_time >= SHUTDOWN_TRIGGER_SECONDS:
                        set_status_flag(status_root, status_label, True, SHUTDOWN_RED_COLOR)
                        debug_print(
                            "shutdown gesture triggered: "
                            f"peace-out gesture held {shutdown_hold_time:.2f}s"
                        )
                        send_quit_keypress()
                        break

                    if shutdown_hold_time >= SHUTDOWN_RED_SECONDS:
                        set_status_flag(status_root, status_label, True, SHUTDOWN_RED_COLOR)
                        new_shutdown_status = "red"
                    elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                        set_status_flag(status_root, status_label, True, SHUTDOWN_YELLOW_COLOR)
                        new_shutdown_status = "yellow"
                    else:
                        set_status_flag(status_root, status_label, True, HAND_FOUND_COLOR)
                        new_shutdown_status = "lime"

                    if new_shutdown_status != shutdown_status:
                        debug_print(
                            "shutdown status changed: "
                            f"{shutdown_status} -> {new_shutdown_status} "
                            f"after {shutdown_hold_time:.2f}s"
                        )
                        shutdown_status = new_shutdown_status

                else:
                    if shutdown_hold_start_time is not None:
                        shutdown_gap_time = (
                            999.0
                            if shutdown_last_seen_time is None
                            else now - shutdown_last_seen_time
                        )

                        if shutdown_gap_time <= SHUTDOWN_GESTURE_GRACE_SECONDS:
                            shutdown_hold_time = now - shutdown_hold_start_time
                            if shutdown_hold_time >= SHUTDOWN_RED_SECONDS:
                                set_status_flag(
                                    status_root, status_label, True, SHUTDOWN_RED_COLOR
                                )
                            elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                                set_status_flag(
                                    status_root, status_label, True, SHUTDOWN_YELLOW_COLOR
                                )
                            else:
                                set_status_flag(
                                    status_root, status_label, True, HAND_FOUND_COLOR
                                )

                            debug_print(
                                "shutdown gesture grace: "
                                f"shutdown gesture briefly not detected "
                                f"({gesture_debug}, "
                                f"gap={shutdown_gap_time:.2f}s)"
                            )
                        else:
                            shutdown_hold_time = now - shutdown_hold_start_time
                            debug_print(
                                "shutdown gesture cancelled: "
                                f"shutdown gesture not detected for {shutdown_gap_time:.2f}s "
                                f"after {shutdown_hold_time:.2f}s"
                            )
                            shutdown_hold_start_time = None
                            shutdown_last_seen_time = None
                            shutdown_status = "none"
                            set_status_flag(status_root, status_label, True, HAND_FOUND_COLOR)
                    else:
                        set_status_flag(status_root, status_label, True, HAND_FOUND_COLOR)

                thumb_dist = gesture_details["thumb_index_distance"]

                shutdown_is_active = shutdown_hold_start_time is not None

                if shutdown_is_active:
                    gesture_state = "shutdown-hold"
                    pinch_was_active = False
                    pinch_start_time = None

                elif thumb_dist <= PINCH_ON_THRESHOLD:
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

                if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                    pinch_hold_time = (
                        0.0 if pinch_start_time is None else now - pinch_start_time
                    )
                    shutdown_hold_time = (
                        0.0
                        if shutdown_hold_start_time is None
                        else now - shutdown_hold_start_time
                    )
                    print(
                        f"hand detected: thumb_dist={thumb_dist:.3f}, "
                        f"pinch_on={PINCH_ON_THRESHOLD:.3f}, "
                        f"release={PINCH_RELEASE_THRESHOLD:.3f}, "
                        f"pinch_hold={pinch_hold_time:.2f}, "
                        f"state={gesture_state}, "
                        f"pinch_was_active={pinch_was_active}, "
                        f"shutdown_gesture={shutdown_gesture}, "
                        f"{gesture_debug}, "
                        f"shutdown_hold={shutdown_hold_time:.2f}, "
                        f"shutdown_status={shutdown_status}"
                    )
                    last_debug_time = now

            else:
                if hand_found_start_time is not None:
                    debug_print("hand-found timer reset: no hand detected")
                hand_found_start_time = None
                hand_found_snapshot_saved = False

                if shutdown_gesture_frame_active:
                    save_debug_frame(rgb_frame, "shutdown_gesture_lost", "no hand landmarks")
                    shutdown_gesture_frame_active = False

                if shutdown_hold_start_time is not None:
                    shutdown_gap_time = (
                        999.0
                        if shutdown_last_seen_time is None
                        else now - shutdown_last_seen_time
                    )

                    if shutdown_gap_time <= SHUTDOWN_GESTURE_GRACE_SECONDS:
                        shutdown_hold_time = now - shutdown_hold_start_time
                        if shutdown_hold_time >= SHUTDOWN_RED_SECONDS:
                            set_status_flag(status_root, status_label, True, SHUTDOWN_RED_COLOR)
                        elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                            set_status_flag(status_root, status_label, True, SHUTDOWN_YELLOW_COLOR)
                        else:
                            set_status_flag(status_root, status_label, True, HAND_FOUND_COLOR)

                        if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                            print(
                                "no hand detected, but shutdown grace is active: "
                                f"gap={shutdown_gap_time:.2f}s, "
                                f"shutdown_hold={shutdown_hold_time:.2f}s, "
                                f"shutdown_status={shutdown_status}"
                            )
                            last_debug_time = now
                    else:
                        shutdown_hold_time = now - shutdown_hold_start_time
                        debug_print(
                            "shutdown gesture cancelled: "
                            f"hand lost for {shutdown_gap_time:.2f}s "
                            f"after {shutdown_hold_time:.2f}s"
                        )
                        shutdown_hold_start_time = None
                        shutdown_last_seen_time = None
                        shutdown_status = "none"
                        set_status_flag(status_root, status_label, False)
                else:
                    set_status_flag(status_root, status_label, False)

                    if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
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
