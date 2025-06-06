#!/usr/bin/env python3
import time
import cv2
from flask import Flask, Response

app = Flask(__name__)

# Path to the constantly‐updated PNG from your C++ app:
FRAME_PATH = "/tmp/output.png"
# Target framerate (match your C++ loop, e.g. 15 FPS):
FPS = 15.0

def generate_mjpeg():
    """
    Generator that reads /tmp/output.png in a loop, encodes it as JPEG,
    and yields a multipart MJPEG frame.
    """
    delay = 1.0 / FPS
    last_frame_time = 0.0

    while True:
        # Throttle to ~FPS
        now = time.time()
        if now - last_frame_time < delay:
            time.sleep(delay - (now - last_frame_time))
            continue
        last_frame_time = time.time()

        img = cv2.imread(FRAME_PATH)
        if img is None:
            # If the C++ program hasn't yet written a frame, wait a bit
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode(".jpg", img)
        if not ret:
            # Failed to encode this frame—skip it
            continue

        frame_bytes = jpeg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/stream")
def stream():
    """
    Flask route: serves multipart/x-mixed-replace MJPEG.
    """
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # Listen on all interfaces, port 8080
    app.run(host="0.0.0.0", port=8080, threaded=True)

