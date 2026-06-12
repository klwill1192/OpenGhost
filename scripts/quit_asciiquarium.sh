#!/usr/bin/env bash
set -e
export DISPLAY="${DISPLAY:-:0.0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
WINDOW_TITLE="asciiquarium-target"
WINDOW_ID=$(xdotool search --name "$WINDOW_TITLE" | head -n 1 || true)
if [ -z "$WINDOW_ID" ]; then
echo "No xterm window found with title: $WINDOW_TITLE"
exit 1
fi
echo "Sending Q to $WINDOW_TITLE ($WINDOW_ID)..."
xdotool key --window "$WINDOW_ID" q
