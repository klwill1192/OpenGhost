#!/usr/bin/env python3

import os
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
# The peace-sign quit gesture should be deliberate and should not be
# confused with thumbs-up. Require index+middle specifically, because
# thumbs-up can make ring/pinky look extended to MediaPipe.
SHUTDOWN_MIN_EXTENDED_FINGERS = 2
SHUTDOWN_MAX_EXTENDED_FINGERS = 3
SHUTDOWN_EXTENDED_FINGER_RATIO = 1.25
SHUTDOWN_MIN_V_SPREAD = 0.10

# Raspberry Pi shutdown gesture tuning.
# Hold a closed fist in view:
# - 0-2 seconds: normal green hand-found flag
# - 2-4 seconds: solid yellow warning flag
# - 4-8 seconds: solid red imminent-shutdown flag
# - 8 seconds: quit the aquarium, then shut down the Raspberry Pi
PI_SHUTDOWN_YELLOW_SECONDS = 2.0
PI_SHUTDOWN_RED_SECONDS = 4.0
PI_SHUTDOWN_TRIGGER_SECONDS = 7.0
PI_SHUTDOWN_MAX_EXTENDED_FINGERS = 1
# Reject thumbs-up so that gesture remains available for a future action.
# A thumb is considered extended when the thumb tip is far from the index
# finger MCP joint relative to the hand's MCP-to-MCP palm width.
PI_SHUTDOWN_REJECT_THUMB_EXTENDED = True
# Thumbs-up was still being accepted with a loose 1.25 ratio.
# Lower this ratio and add a wrist-based thumb check below so an extended
# thumb is rejected more reliably while a folded thumb can still be accepted
# as a closed fist.
PI_SHUTDOWN_THUMB_EXTENDED_RATIO = 0.85
# A real closed fist in testing had thumb/index distance around 0.14-0.15.
# Thumbs-up should normally move the thumb farther away, so reject only when
# the thumb/index distance is clearly larger than a closed-fist value.
PI_SHUTDOWN_MAX_THUMB_INDEX_DISTANCE = 0.22
PI_SHUTDOWN_THUMB_WRIST_RATIO = 1.20

# Happy Fish gesture tuning.
# Hold a thumbs-up gesture briefly to send H to the aquarium.
HAPPY_FISH_HOLD_SECONDS = 2.0
HAPPY_FISH_GESTURE_GRACE_SECONDS = 0.50
HAPPY_FISH_STATUS_SECONDS = 8.0
HAPPY_FISH_THUMB_EXTENDED_RATIO = 0.85
HAPPY_FISH_MIN_THUMB_WRIST_RATIO = 1.15
HAPPY_FISH_MAX_EXTENDED_FINGERS = 2

# Keypress target
KEY_TO_SEND = "f"
QUIT_KEY_TO_SEND = "q"
HAPPY_KEY_TO_SEND = "h"
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
# These defaults match the current 50mm beam-splitter profile.
# The launcher can override them with environment variables for other
# display/beam-splitter profiles.
def env_int(name: str, default: int) -> int:
    """Return an integer environment override, or the default if unset/invalid."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"WARNING: ignoring invalid {name}={raw_value!r}; using {default}")
        return default


STATUS_WINDOW_X = env_int("STATUS_WINDOW_X", 510)
STATUS_WINDOW_Y = env_int("STATUS_WINDOW_Y", 430)
STATUS_WINDOW_WIDTH = env_int("STATUS_WINDOW_WIDTH", 40)
STATUS_WINDOW_HEIGHT = env_int("STATUS_WINDOW_HEIGHT", 40)
STATUS_FLAG_FONT_SIZE = env_int("STATUS_FLAG_FONT_SIZE", 24)

NO_HAND_FLAG = "⚑"
HAND_FLAG = "⚑"

NO_HAND_COLOR = "white"
HAND_FOUND_COLOR = "lime"
SHUTDOWN_YELLOW_COLOR = "yellow"
SHUTDOWN_RED_COLOR = "red"
HAPPY_FISH_COLOR = "magenta"
HAPPY_FISH_STATUS_UNTIL = 0.0


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


def send_key_to_target(key: str, label: str, also_send_to_focused_window: bool = False) -> bool:
    """Send a key to the target xterm window using xdotool.

    Feed should send exactly one targeted key event. Quit uses the optional
    focused-window key event as an extra reliability measure.
    """
    print(f"Sending {label} key")

    sent = False

    for window_id in find_target_windows():
        debug_print(f"Sending {label} to window {window_id}")

        # Activating first is more reliable for terminal applications than
        # relying only on xdotool's --window argument.
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
        if also_send_to_focused_window:
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
    """Send one feed key to the target xterm window."""
    send_key_to_target(KEY_TO_SEND, KEY_TO_SEND.upper())


def send_happy_fish_keypress() -> None:
    """Send H to the target xterm window to start Happy Fish mode."""
    send_key_to_target(HAPPY_KEY_TO_SEND, HAPPY_KEY_TO_SEND.upper())


def send_quit_keypress() -> None:
    """Send q to the aquarium, then close the xterm if q did not exit it."""
    if send_key_to_target(
        QUIT_KEY_TO_SEND,
        QUIT_KEY_TO_SEND.upper(),
        also_send_to_focused_window=True,
    ):
        close_target_windows_after_quit_grace()


def shutdown_raspberry_pi() -> bool:
    """Request a Raspberry Pi OS shutdown without waiting on this process.

    Start the shutdown command in its own session before we quit the aquarium.
    That way, if the launcher notices the aquarium xterm closing and stops this
    OpenGhost process, the OS shutdown request has already been handed to sudo.
    """
    print("Requesting Raspberry Pi shutdown with sudo -n /usr/sbin/shutdown -h now")
    try:
        subprocess.Popen(
            ["sudo", "-n", "/usr/sbin/shutdown", "-h", "now"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        print(f"WARNING: failed to request Raspberry Pi shutdown: {exc}")
        return False


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
        font=("DejaVu Sans", STATUS_FLAG_FONT_SIZE, "bold"),
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


def set_status_flag(root, label, flag_color: str) -> None:
    """Update the overlay flag.

    Happy Fish mode is a timed status that should survive brief hand loss.
    Let yellow/red shutdown warnings override it, but prevent normal green
    or white/no-hand updates from overwriting magenta while the Happy Fish
    status timer is active.
    """
    if (
        time.time() < HAPPY_FISH_STATUS_UNTIL
        and flag_color not in (SHUTDOWN_YELLOW_COLOR, SHUTDOWN_RED_COLOR)
    ):
        flag_color = HAPPY_FISH_COLOR

    if flag_color == NO_HAND_COLOR:
        label.config(text=NO_HAND_FLAG, fg=NO_HAND_COLOR)
    else:
        label.config(text=HAND_FLAG, fg=flag_color)

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
    - require index and middle fingers specifically;
    - require the index and middle fingertips to be spread apart;
    - reject a full open palm by requiring that not all four fingers are
      counted as extended.

    Earlier versions accepted any two extended non-thumb fingers. That worked
    for some angled peace signs, but it also let thumbs-up trigger aquarium
    quit when MediaPipe interpreted ring/pinky as extended.
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

    index_extended = "index" in extended_fingers
    middle_extended = "middle" in extended_fingers
    index_middle_spread = landmark_distance(
        hand_landmarks,
        HandLandmark.INDEX_FINGER_TIP,
        HandLandmark.MIDDLE_FINGER_TIP,
    )
    spread_ok = index_middle_spread >= SHUTDOWN_MIN_V_SPREAD
    index_middle_ok = index_extended and middle_extended

    shutdown_gesture = index_middle_ok and not_open_palm and spread_ok

    details = {
        "ratios": ratios,
        "thumb_index_distance": thumb_index_distance,
        "extended_fingers": extended_fingers,
        "extended_count": extended_count,
        "best_pair": best_pair,
        "best_pair_spread": best_pair_spread,
        "index_middle_spread": index_middle_spread,
        "enough_extended": enough_extended,
        "index_extended": index_extended,
        "middle_extended": middle_extended,
        "index_middle_ok": index_middle_ok,
        "not_open_palm": not_open_palm,
        "spread_ok": spread_ok,
    }
    return shutdown_gesture, details


def evaluate_pi_shutdown_gesture(hand_landmarks, gesture_details):
    """Return Raspberry Pi shutdown-gesture decision plus debug data.

    Raspberry Pi shutdown gesture rules:
    - use a deliberately held closed fist;
    - count the same non-thumb finger-extension ratios used for the peace sign;
    - treat the hand as a fist when no more than one non-thumb finger appears
      extended.

    The long 8-second hold requirement is the main safety guard.
    """
    extended_count = gesture_details["extended_count"]
    extended_fingers = gesture_details["extended_fingers"]
    ratios = gesture_details["ratios"]
    thumb_index_distance = gesture_details["thumb_index_distance"]

    palm_width = landmark_distance(
        hand_landmarks,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    )
    thumb_tip_to_index_mcp = landmark_distance(
        hand_landmarks,
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_FINGER_MCP,
    )
    wrist_to_thumb_tip = landmark_distance(
        hand_landmarks,
        HandLandmark.WRIST,
        HandLandmark.THUMB_TIP,
    )
    wrist_to_thumb_mcp = landmark_distance(
        hand_landmarks,
        HandLandmark.WRIST,
        HandLandmark.THUMB_MCP,
    )

    if palm_width <= 0.0001:
        thumb_extension_ratio = 0.0
    else:
        thumb_extension_ratio = thumb_tip_to_index_mcp / palm_width

    if wrist_to_thumb_mcp <= 0.0001:
        thumb_wrist_ratio = 0.0
    else:
        thumb_wrist_ratio = wrist_to_thumb_tip / wrist_to_thumb_mcp

    thumb_extended_by_index = (
        thumb_extension_ratio >= PI_SHUTDOWN_THUMB_EXTENDED_RATIO
    )
    # Keep the wrist ratio only as a diagnostic. It was too aggressive as a
    # rejection test because a real closed fist can still put the thumb tip
    # farther from the wrist than the thumb MCP.
    thumb_extended_by_wrist = False
    thumb_too_far_from_index = (
        thumb_index_distance > PI_SHUTDOWN_MAX_THUMB_INDEX_DISTANCE
    )
    thumb_extended = thumb_extended_by_index or thumb_too_far_from_index

    few_non_thumb_fingers_extended = (
        extended_count <= PI_SHUTDOWN_MAX_EXTENDED_FINGERS
    )
    thumb_ok = (not PI_SHUTDOWN_REJECT_THUMB_EXTENDED) or (not thumb_extended)
    closed_fist = few_non_thumb_fingers_extended and thumb_ok

    details = {
        "ratios": ratios,
        "extended_fingers": extended_fingers,
        "extended_count": extended_count,
        "palm_width": palm_width,
        "thumb_tip_to_index_mcp": thumb_tip_to_index_mcp,
        "wrist_to_thumb_tip": wrist_to_thumb_tip,
        "wrist_to_thumb_mcp": wrist_to_thumb_mcp,
        "thumb_index_distance": thumb_index_distance,
        "thumb_extension_ratio": thumb_extension_ratio,
        "thumb_wrist_ratio": thumb_wrist_ratio,
        "thumb_extended_by_index": thumb_extended_by_index,
        "thumb_extended_by_wrist": thumb_extended_by_wrist,
        "thumb_too_far_from_index": thumb_too_far_from_index,
        "thumb_extended": thumb_extended,
        "few_non_thumb_fingers_extended": few_non_thumb_fingers_extended,
        "thumb_ok": thumb_ok,
        "closed_fist": closed_fist,
    }
    return closed_fist, details


def evaluate_thumbs_up_gesture(hand_landmarks, gesture_details):
    """Return thumbs-up gesture decision plus debug data.

    A thumbs-up should have the thumb extended while the non-thumb
    fingers are mostly curled. This keeps it separate from the
    peace-sign quit gesture and the closed-fist Pi shutdown gesture.
    """
    extended_count = gesture_details["extended_count"]
    extended_fingers = gesture_details["extended_fingers"]
    ratios = gesture_details["ratios"]
    thumb_index_distance = gesture_details["thumb_index_distance"]
    palm_width = landmark_distance(
        hand_landmarks,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    )
    thumb_tip_to_index_mcp = landmark_distance(
        hand_landmarks,
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_FINGER_MCP,
    )
    wrist_to_thumb_tip = landmark_distance(
        hand_landmarks,
        HandLandmark.WRIST,
        HandLandmark.THUMB_TIP,
    )
    wrist_to_thumb_mcp = landmark_distance(
        hand_landmarks,
        HandLandmark.WRIST,
        HandLandmark.THUMB_MCP,
    )
    if palm_width <= 0.0001:
        thumb_extension_ratio = 0.0
    else:
        thumb_extension_ratio = thumb_tip_to_index_mcp / palm_width
    if wrist_to_thumb_mcp <= 0.0001:
        thumb_wrist_ratio = 0.0
    else:
        thumb_wrist_ratio = wrist_to_thumb_tip / wrist_to_thumb_mcp
    thumb_extended = (
        thumb_extension_ratio >= HAPPY_FISH_THUMB_EXTENDED_RATIO
        or thumb_wrist_ratio >= HAPPY_FISH_MIN_THUMB_WRIST_RATIO
    )
    few_non_thumb_fingers_extended = (
        extended_count <= HAPPY_FISH_MAX_EXTENDED_FINGERS
    )
    index_extended = "index" in extended_fingers
    middle_extended = "middle" in extended_fingers
    peace_like = index_extended and middle_extended

    thumbs_up = (
        thumb_extended
        and few_non_thumb_fingers_extended
        and not peace_like
    )
    details = {
        "ratios": ratios,
        "extended_fingers": extended_fingers,
        "extended_count": extended_count,
        "thumb_index_distance": thumb_index_distance,
        "palm_width": palm_width,
        "thumb_tip_to_index_mcp": thumb_tip_to_index_mcp,
        "wrist_to_thumb_tip": wrist_to_thumb_tip,
        "wrist_to_thumb_mcp": wrist_to_thumb_mcp,
        "thumb_extension_ratio": thumb_extension_ratio,
        "thumb_wrist_ratio": thumb_wrist_ratio,
        "thumb_extended": thumb_extended,
        "few_non_thumb_fingers_extended": few_non_thumb_fingers_extended,
        "index_extended": index_extended,
        "middle_extended": middle_extended,
        "peace_like": peace_like,
        "thumbs_up": thumbs_up,
    }
    return thumbs_up, details


def format_thumbs_up_debug(details) -> str:
    """Format thumbs-up gesture measurements."""
    ratios = details["ratios"]
    extended_text = ",".join(details["extended_fingers"]) or "none"
    return (
        f"thumbs_extended={details['extended_count']}/4 [{extended_text}], "
        f"index_ratio={ratios['index']:.2f}, "
        f"middle_ratio={ratios['middle']:.2f}, "
        f"ring_ratio={ratios['ring']:.2f}, "
        f"pinky_ratio={ratios['pinky']:.2f}, "
        f"thumb_index={details['thumb_index_distance']:.3f}, "
        f"thumb_ratio={details['thumb_extension_ratio']:.2f}, "
        f"thumb_wrist_ratio={details['thumb_wrist_ratio']:.2f}, "
        f"thumb_extended:{details['thumb_extended']}, "
        f"few_fingers:{details['few_non_thumb_fingers_extended']}, "
        f"index_extended:{details['index_extended']}, "
        f"middle_extended:{details['middle_extended']}, "
        f"peace_like:{details['peace_like']}, "
        f"thumbs_up:{details['thumbs_up']}"
    )


def format_pi_shutdown_debug(details) -> str:
    """Format Raspberry Pi shutdown-gesture measurements."""
    ratios = details["ratios"]
    extended_text = ",".join(details["extended_fingers"]) or "none"
    return (
        f"fist_extended={details['extended_count']}/4 [{extended_text}], "
        f"index_ratio={ratios['index']:.2f}, "
        f"middle_ratio={ratios['middle']:.2f}, "
        f"ring_ratio={ratios['ring']:.2f}, "
        f"pinky_ratio={ratios['pinky']:.2f}, "
        f"thumb_index={details['thumb_index_distance']:.3f}, "
        f"thumb_ratio={details['thumb_extension_ratio']:.2f}, "
        f"thumb_wrist_ratio={details['thumb_wrist_ratio']:.2f}, "
        f"thumb_ext_index:{details['thumb_extended_by_index']}, "
        f"thumb_too_far:{details['thumb_too_far_from_index']}, "
        f"thumb_extended:{details['thumb_extended']}, "
        f"few_fingers:{details['few_non_thumb_fingers_extended']}, "
        f"thumb_ok:{details['thumb_ok']}, "
        f"closed_fist:{details['closed_fist']}"
    )


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
        f"index_middle_spread={details.get('index_middle_spread', 0.0):.3f}, "
        f"index_ratio={ratios['index']:.2f}, "
        f"middle_ratio={ratios['middle']:.2f}, "
        f"ring_ratio={ratios['ring']:.2f}, "
        f"pinky_ratio={ratios['pinky']:.2f}, "
        f"checks=index:{details.get('index_extended', False)} "
        f"middle:{details.get('middle_extended', False)} "
        f"index_middle:{details.get('index_middle_ok', False)} "
        f"not_open_palm:{details['not_open_palm']} "
        f"spread:{details['spread_ok']}"
    )

def main() -> None:
    global HAPPY_FISH_STATUS_UNTIL

    pinch_was_active = False
    pinch_start_time = None
    last_keypress_time = 0.0
    last_debug_time = 0.0

    shutdown_hold_start_time = None
    shutdown_last_seen_time = None
    shutdown_status = "none"
    shutdown_gesture_frame_active = False

    pi_shutdown_hold_start_time = None
    pi_shutdown_last_seen_time = None
    pi_shutdown_status = "none"
    pi_shutdown_gesture_frame_active = False
    happy_fish_hold_start_time = None
    happy_fish_last_seen_time = None
    happy_fish_last_keypress_time = 0.0
    happy_fish_mode_until = 0.0
    happy_fish_sent_for_hold = False

    hand_found_start_time = None
    hand_found_snapshot_saved = False

    def shutdown_warning_flag_is_active(current_time: float) -> bool:
        # Return True when shutdown warning colors should override Happy Fish.
        if pi_shutdown_hold_start_time is not None:
            return (
                current_time - pi_shutdown_hold_start_time
                >= PI_SHUTDOWN_YELLOW_SECONDS
            )

        if shutdown_hold_start_time is not None:
            return (
                current_time - shutdown_hold_start_time
                >= SHUTDOWN_YELLOW_SECONDS
            )

        return False

    def happy_fish_status_is_active(current_time: float) -> bool:
        # Return True while the OpenGhost Happy Fish status timer is active.
        return current_time < happy_fish_mode_until

    def refresh_happy_fish_status_flag(current_time: float) -> None:
        # Keep the flag magenta during Happy Fish mode unless shutdown warning wins.
        if (
            happy_fish_status_is_active(current_time)
            and not shutdown_warning_flag_is_active(current_time)
        ):
            set_status_flag(status_root, status_label, HAPPY_FISH_COLOR)

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
    print(f"Happy Fish key to send: {HAPPY_KEY_TO_SEND}")
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
    print(f"Pi shutdown yellow threshold: {PI_SHUTDOWN_YELLOW_SECONDS}")
    print(f"Pi shutdown red threshold: {PI_SHUTDOWN_RED_SECONDS}")
    print(f"Pi shutdown trigger threshold: {PI_SHUTDOWN_TRIGGER_SECONDS}")
    print(f"Pi shutdown max extended fingers: {PI_SHUTDOWN_MAX_EXTENDED_FINGERS}")
    print(f"Pi shutdown reject thumb extended: {PI_SHUTDOWN_REJECT_THUMB_EXTENDED}")
    print(f"Pi shutdown thumb extended ratio: {PI_SHUTDOWN_THUMB_EXTENDED_RATIO}")
    print(f"Happy Fish thumbs-up hold time: {HAPPY_FISH_HOLD_SECONDS}")
    print(f"Happy Fish gesture grace period: {HAPPY_FISH_GESTURE_GRACE_SECONDS}")
    print(f"Happy Fish status duration: {HAPPY_FISH_STATUS_SECONDS}")
    print(f"Happy Fish status flag color: {HAPPY_FISH_COLOR}")
    print(f"Happy Fish thumb extended ratio: {HAPPY_FISH_THUMB_EXTENDED_RATIO}")
    print(f"Happy Fish thumb/wrist ratio: {HAPPY_FISH_MIN_THUMB_WRIST_RATIO}")
    print(f"Happy Fish max extended fingers: {HAPPY_FISH_MAX_EXTENDED_FINGERS}")
    print(
        "Pi shutdown max thumb/index distance: "
        f"{PI_SHUTDOWN_MAX_THUMB_INDEX_DISTANCE}"
    )
    print(
        "Status flag overlay: "
        f"x={STATUS_WINDOW_X}, y={STATUS_WINDOW_Y}, "
        f"width={STATUS_WINDOW_WIDTH}, height={STATUS_WINDOW_HEIGHT}, "
        f"font_size={STATUS_FLAG_FONT_SIZE}"
    )
    if DEBUG_CAPTURE_FRAMES:
        print(f"Debug frames will be saved in: {DEBUG_FRAME_DIR}")
        print(
            "A hand-found debug frame will be saved after "
            f"{DEBUG_HAND_FOUND_SNAPSHOT_SECONDS:.1f}s of continuous hand detection."
        )
    print("Pinch thumb/index finger, then release, to send F.")
    print("Hold a two-finger V / peace-out gesture for 5 seconds to quit the aquarium.")
    print("Hold a closed fist for 8 seconds to shut down the Raspberry Pi.")
    print("Hold a thumbs-up gesture for 2 seconds to start Happy Fish mode.")
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
                pi_shutdown_gesture, pi_gesture_details = evaluate_pi_shutdown_gesture(
                    hand_landmarks, gesture_details
                )
                pi_gesture_debug = format_pi_shutdown_debug(pi_gesture_details)
                thumbs_up_gesture, thumbs_up_details = evaluate_thumbs_up_gesture(
                    hand_landmarks, gesture_details
                )
                thumbs_up_debug = format_thumbs_up_debug(thumbs_up_details)

                if (
                    shutdown_gesture
                    or shutdown_hold_start_time is not None
                    or pi_shutdown_gesture
                    or pi_shutdown_hold_start_time is not None
                ):
                    # Shutdown gestures have priority.  A peace sign can look
                    # thumb-extended in some MediaPipe frames, so do not let it
                    # also start Happy Fish mode.
                    thumbs_up_gesture = False

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

                if pi_shutdown_gesture and not pi_shutdown_gesture_frame_active:
                    save_debug_frame(
                        rgb_frame,
                        "pi_shutdown_gesture_detected",
                        pi_gesture_debug,
                        hand_landmarks,
                    )
                    pi_shutdown_gesture_frame_active = True
                elif not pi_shutdown_gesture and pi_shutdown_gesture_frame_active:
                    save_debug_frame(
                        rgb_frame,
                        "pi_shutdown_gesture_lost",
                        pi_gesture_debug,
                        hand_landmarks,
                    )
                    pi_shutdown_gesture_frame_active = False

                if pi_shutdown_gesture:
                    pi_shutdown_last_seen_time = now

                    if pi_shutdown_hold_start_time is None:
                        pi_shutdown_hold_start_time = now
                        pi_shutdown_status = "lime"
                        debug_print(
                            "pi shutdown gesture started: "
                            f"closed fist detected ({pi_gesture_debug})"
                        )

                    pi_shutdown_hold_time = now - pi_shutdown_hold_start_time

                    if pi_shutdown_hold_time >= PI_SHUTDOWN_TRIGGER_SECONDS:
                        set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        debug_print(
                            "pi shutdown gesture triggered: "
                            f"closed fist held {pi_shutdown_hold_time:.2f}s"
                        )
                        # Request OS shutdown before quitting the aquarium.
                        # Quitting the aquarium can cause the launcher to stop
                        # this OpenGhost process immediately, so the OS shutdown
                        # request must be started first.
                        shutdown_raspberry_pi()
                        send_quit_keypress()
                        break

                    if pi_shutdown_hold_time >= PI_SHUTDOWN_RED_SECONDS:
                        set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        new_pi_shutdown_status = "red"
                    elif pi_shutdown_hold_time >= PI_SHUTDOWN_YELLOW_SECONDS:
                        set_status_flag(status_root, status_label, SHUTDOWN_YELLOW_COLOR)
                        new_pi_shutdown_status = "yellow"
                    else:
                        set_status_flag(status_root, status_label, HAND_FOUND_COLOR)
                        new_pi_shutdown_status = "lime"

                    if new_pi_shutdown_status != pi_shutdown_status:
                        debug_print(
                            "pi shutdown status changed: "
                            f"{pi_shutdown_status} -> {new_pi_shutdown_status} "
                            f"after {pi_shutdown_hold_time:.2f}s"
                        )
                        pi_shutdown_status = new_pi_shutdown_status

                        if new_pi_shutdown_status in ("yellow", "red"):
                            if now < happy_fish_mode_until or happy_fish_hold_start_time is not None:
                                debug_print(
                                    "Happy Fish mode cancelled: Pi shutdown sequence reached "
                                    f"{new_pi_shutdown_status} status"
                                )
                            happy_fish_mode_until = 0.0
                            happy_fish_hold_start_time = None
                            happy_fish_sent_for_hold = False

                    # While the Pi shutdown gesture is active, do not also treat
                    # the hand as a peace-sign aquarium-quit gesture.
                    shutdown_hold_start_time = None
                    shutdown_last_seen_time = None
                    shutdown_status = "none"

                elif pi_shutdown_hold_start_time is not None:
                    # A Raspberry Pi OS shutdown is a high-impact action.
                    # Do not keep counting through a visible non-fist hand shape,
                    # such as thumbs-up.  Grace is only allowed in the no-hand
                    # branch below, where MediaPipe may have briefly lost the hand.
                    pi_shutdown_hold_time = now - pi_shutdown_hold_start_time
                    debug_print(
                        "pi shutdown gesture cancelled: "
                        f"visible hand is no longer a closed fist "
                        f"after {pi_shutdown_hold_time:.2f}s "
                        f"({pi_gesture_debug})"
                    )
                    pi_shutdown_hold_start_time = None
                    pi_shutdown_last_seen_time = None
                    pi_shutdown_status = "none"
                    set_status_flag(status_root, status_label, HAND_FOUND_COLOR)

                elif shutdown_gesture:
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
                        set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        debug_print(
                            "shutdown gesture triggered: "
                            f"peace-out gesture held {shutdown_hold_time:.2f}s"
                        )
                        send_quit_keypress()
                        break

                    if shutdown_hold_time >= SHUTDOWN_RED_SECONDS:
                        set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        new_shutdown_status = "red"
                    elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                        set_status_flag(status_root, status_label, SHUTDOWN_YELLOW_COLOR)
                        new_shutdown_status = "yellow"
                    else:
                        set_status_flag(status_root, status_label, HAND_FOUND_COLOR)
                        new_shutdown_status = "lime"

                    if new_shutdown_status != shutdown_status:
                        debug_print(
                            "shutdown status changed: "
                            f"{shutdown_status} -> {new_shutdown_status} "
                            f"after {shutdown_hold_time:.2f}s"
                        )
                        shutdown_status = new_shutdown_status

                        if new_shutdown_status in ("yellow", "red"):
                            if now < happy_fish_mode_until or happy_fish_hold_start_time is not None:
                                debug_print(
                                    "Happy Fish mode cancelled: aquarium shutdown sequence reached "
                                    f"{new_shutdown_status} status"
                                )
                            happy_fish_mode_until = 0.0
                            happy_fish_hold_start_time = None
                            happy_fish_sent_for_hold = False

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
                                    status_root, status_label, SHUTDOWN_RED_COLOR
                                )
                            elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                                set_status_flag(
                                    status_root, status_label, SHUTDOWN_YELLOW_COLOR
                                )
                            else:
                                set_status_flag(
                                    status_root, status_label, HAND_FOUND_COLOR
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
                            set_status_flag(status_root, status_label, HAND_FOUND_COLOR)
                    else:
                        set_status_flag(status_root, status_label, HAND_FOUND_COLOR)

                if now < happy_fish_mode_until:
                    set_status_flag(status_root, status_label, HAPPY_FISH_COLOR)

                if thumbs_up_gesture:
                    happy_fish_last_seen_time = now

                    if happy_fish_hold_start_time is None:
                        happy_fish_hold_start_time = now
                        happy_fish_sent_for_hold = False
                        debug_print(
                            "Happy Fish thumbs-up gesture started: "
                            f"{thumbs_up_debug}"
                        )
                    happy_fish_hold_time = now - happy_fish_hold_start_time
                    if (
                        happy_fish_hold_time >= HAPPY_FISH_HOLD_SECONDS
                        and not happy_fish_sent_for_hold
                        and now - happy_fish_last_keypress_time >= KEYPRESS_COOLDOWN_SECONDS
                    ):
                        debug_print(
                            "Happy Fish thumbs-up gesture triggered: "
                            f"held {happy_fish_hold_time:.2f}s"
                        )
                        send_happy_fish_keypress()
                        happy_fish_last_keypress_time = now
                        happy_fish_mode_until = now + HAPPY_FISH_STATUS_SECONDS
                        HAPPY_FISH_STATUS_UNTIL = happy_fish_mode_until
                        happy_fish_sent_for_hold = True
                        set_status_flag(status_root, status_label, HAPPY_FISH_COLOR)
                    elif now < happy_fish_mode_until:
                        set_status_flag(status_root, status_label, HAPPY_FISH_COLOR)
                    else:
                        set_status_flag(status_root, status_label, HAND_FOUND_COLOR)
                    pinch_was_active = False
                    pinch_start_time = None
                    if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                        print(
                            "\n" + f"hand detected: thumbs_up_hold={happy_fish_hold_time:.2f}, "
                            f"thumbs_up_gesture={thumbs_up_gesture}, "
                            f"happy_fish_active={now < happy_fish_mode_until}, "
                            f"{thumbs_up_debug}"
                        )
                        last_debug_time = now
                    refresh_happy_fish_status_flag(now)
                    time.sleep(0.01)
                    continue
                elif happy_fish_hold_start_time is not None:
                    happy_fish_gap_time = (
                        999.0
                        if happy_fish_last_seen_time is None
                        else now - happy_fish_last_seen_time
                    )

                    if happy_fish_gap_time <= HAPPY_FISH_GESTURE_GRACE_SECONDS:
                        happy_fish_hold_time = now - happy_fish_hold_start_time
                        if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                            print(
                                "\nHappy Fish thumbs-up gesture grace: "
                                f"gap={happy_fish_gap_time:.2f}s, "
                                f"hold={happy_fish_hold_time:.2f}s"
                            )
                            last_debug_time = now
                    else:
                        debug_print(
                            "Happy Fish thumbs-up gesture cancelled: "
                            f"gap={happy_fish_gap_time:.2f}s"
                        )
                        happy_fish_hold_start_time = None
                        happy_fish_last_seen_time = None
                        happy_fish_sent_for_hold = False

                thumb_dist = gesture_details["thumb_index_distance"]

                shutdown_is_active = (
                    shutdown_hold_start_time is not None
                    or pi_shutdown_hold_start_time is not None
                )

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
                        "\n" + f"hand detected: thumb_dist={thumb_dist:.3f}, "
                        f"pinch_on={PINCH_ON_THRESHOLD:.3f}, "
                        f"release={PINCH_RELEASE_THRESHOLD:.3f}, "
                        f"pinch_hold={pinch_hold_time:.2f}, "
                        f"state={gesture_state}, "
                        f"pinch_was_active={pinch_was_active}, "
                        f"shutdown_gesture={shutdown_gesture}, "
                        f"{gesture_debug}, "
                        f"pi_shutdown_gesture={pi_shutdown_gesture}, "
                        f"{pi_gesture_debug}, "
                        f"shutdown_hold={shutdown_hold_time:.2f}, "
                        f"shutdown_status={shutdown_status}, "
                        f"pi_shutdown_status={pi_shutdown_status}"
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

                if pi_shutdown_hold_start_time is not None:
                    pi_shutdown_gap_time = (
                        999.0
                        if pi_shutdown_last_seen_time is None
                        else now - pi_shutdown_last_seen_time
                    )

                    if pi_shutdown_gap_time <= SHUTDOWN_GESTURE_GRACE_SECONDS:
                        pi_shutdown_hold_time = now - pi_shutdown_hold_start_time
                        if pi_shutdown_hold_time >= PI_SHUTDOWN_RED_SECONDS:
                            set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        elif pi_shutdown_hold_time >= PI_SHUTDOWN_YELLOW_SECONDS:
                            set_status_flag(status_root, status_label, SHUTDOWN_YELLOW_COLOR)
                        else:
                            set_status_flag(status_root, status_label, HAND_FOUND_COLOR)

                        if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                            print(
                                "\nno hand detected, but pi shutdown grace is active: "
                                f"gap={pi_shutdown_gap_time:.2f}s, "
                                f"pi_shutdown_hold={pi_shutdown_hold_time:.2f}s, "
                                f"pi_shutdown_status={pi_shutdown_status}"
                            )
                            last_debug_time = now
                    else:
                        pi_shutdown_hold_time = now - pi_shutdown_hold_start_time
                        debug_print(
                            "pi shutdown gesture cancelled: "
                            f"hand lost for {pi_shutdown_gap_time:.2f}s "
                            f"after {pi_shutdown_hold_time:.2f}s"
                        )
                        pi_shutdown_hold_start_time = None
                        pi_shutdown_last_seen_time = None
                        pi_shutdown_status = "none"
                        set_status_flag(status_root, status_label, NO_HAND_COLOR)

                elif shutdown_hold_start_time is not None:
                    shutdown_gap_time = (
                        999.0
                        if shutdown_last_seen_time is None
                        else now - shutdown_last_seen_time
                    )

                    if shutdown_gap_time <= SHUTDOWN_GESTURE_GRACE_SECONDS:
                        shutdown_hold_time = now - shutdown_hold_start_time
                        if shutdown_hold_time >= SHUTDOWN_RED_SECONDS:
                            set_status_flag(status_root, status_label, SHUTDOWN_RED_COLOR)
                        elif shutdown_hold_time >= SHUTDOWN_YELLOW_SECONDS:
                            set_status_flag(status_root, status_label, SHUTDOWN_YELLOW_COLOR)
                        else:
                            set_status_flag(status_root, status_label, HAND_FOUND_COLOR)

                        if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                            print(
                                "\nno hand detected, but shutdown grace is active: "
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
                        set_status_flag(status_root, status_label, NO_HAND_COLOR)
                else:
                    set_status_flag(status_root, status_label, NO_HAND_COLOR)

                    if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                        print("\nno hand detected")
                        last_debug_time = now

                if happy_fish_hold_start_time is not None:
                    happy_fish_gap_time = (
                        999.0
                        if happy_fish_last_seen_time is None
                        else now - happy_fish_last_seen_time
                    )

                    if happy_fish_gap_time <= HAPPY_FISH_GESTURE_GRACE_SECONDS:
                        happy_fish_hold_time = now - happy_fish_hold_start_time
                        if DEBUG_MESSAGES and now - last_debug_time >= DEBUG_INTERVAL_SECONDS:
                            print(
                                "\nno hand detected, but Happy Fish thumbs-up grace is active: "
                                f"gap={happy_fish_gap_time:.2f}s, "
                                f"hold={happy_fish_hold_time:.2f}s"
                            )
                            last_debug_time = now
                    else:
                        debug_print(
                            "Happy Fish thumbs-up gesture cancelled: no hand detected "
                            f"for {happy_fish_gap_time:.2f}s"
                        )
                        happy_fish_hold_start_time = None
                        happy_fish_last_seen_time = None
                        happy_fish_sent_for_hold = False
                pinch_was_active = False
                pinch_start_time = None

            refresh_happy_fish_status_flag(now)
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
