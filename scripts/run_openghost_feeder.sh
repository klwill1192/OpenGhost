#!/usr/bin/env bash
set -e
# Use the Raspberry Pi desktop display.
# If running from SSH while the desktop is active, these are usually needed.
export DISPLAY="${DISPLAY:-:0.0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
# py5 requires Java 17 and must not run in headless mode.
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
export PATH="$JAVA_HOME/bin:$PATH"
# Make sure Java is not being forced into headless mode.
unset JAVA_TOOL_OPTIONS
unset _JAVA_OPTIONS
OPENGHOST_DIR="$HOME/OpenGhost"
VENV_DIR="$OPENGHOST_DIR/.venv"
WINDOW_TITLE="asciiquarium-target"
source "$VENV_DIR/bin/activate"
cd "$OPENGHOST_DIR"
echo "Using DISPLAY=$DISPLAY"
echo "Using XAUTHORITY=$XAUTHORITY"
echo "Using JAVA_HOME=$JAVA_HOME"
echo "Starting feeding asciiquarium in xterm..."
xterm \
-T "$WINDOW_TITLE" \
-geometry 86x40+102+102 \
-bg black \
-fg white \
-e bash -lc "source '$VENV_DIR/bin/activate'; python -m asciiquarium.main" &
AQUARIUM_PID=$!
sleep 2
if xdotool search --name "$WINDOW_TITLE" >/dev/null 2>&1; then
echo "Found xterm window: $WINDOW_TITLE"
else
echo "WARNING: xdotool could not find window named $WINDOW_TITLE"
echo "Make sure the desktop is running and xterm opened correctly."
fi
echo "Starting OpenGhost finger tracking..."

python pinch_release_keypress.py &
OPENGHOST_PID=$!

# Give the OpenGhost/py5 window time to appear, then bring asciiquarium back to front.
sleep 4

AQUARIUM_WINDOW_ID=$(xdotool search --name "$WINDOW_TITLE" | head -n 1 || true)

if [ -n "$AQUARIUM_WINDOW_ID" ]; then
    echo "Raising asciiquarium window: $AQUARIUM_WINDOW_ID"
    xdotool windowactivate "$AQUARIUM_WINDOW_ID"
    xdotool windowraise "$AQUARIUM_WINDOW_ID"
else
    echo "WARNING: could not raise asciiquarium window named $WINDOW_TITLE"
fi
cleanup() {
echo "Stopping OpenGhost..."
kill "$OPENGHOST_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
# Keep the script alive until the aquarium xterm exits.
wait "$AQUARIUM_PID"
