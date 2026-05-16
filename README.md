# Image Behaviour Alerts

Python utilities to capture RTSP images and monitor a camera for a user-selected
object with local YOLO detection.

## Desktop interface

Run the app to open the desktop interface:

```powershell
uv run python main.py
```

The window uses the `.env` defaults. It can start and stop object monitoring,
capture one image, display the latest camera frame, and show status and alert
messages.

The default model downloads on first use if it is not already cached. Standard
COCO labels include `person`, `car`, `dog`, `cat`, `cell phone`, and `bottle`.

## Type checking

First-party Python code should not introduce `Any` or `object` annotations. Use
explicit types, local Protocols, or narrow type aliases for dynamic library
boundaries.

```powershell
uv run pyright
uv run python -m unittest discover -s tests
```
