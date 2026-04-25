# AI Human Webcam Detector

This app uses your webcam and a pretrained YOLO model to detect whether a human is visible. The live camera window overlays:

- Human / no-human status
- Prediction percentage
- Person bounding boxes
- FPS

## Run on Windows PowerShell

From this folder:

```powershell
cd "C:\Users\11 TRDs\Downloads\interactive\human-webcam-ai"
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python human_detector.py
```

If PowerShell blocks activation, run the app directly through the virtual environment:

```powershell
cd "C:\Users\11 TRDs\Downloads\interactive\human-webcam-ai"
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe human_detector.py
```

The first run may take longer because YOLO downloads the model file.

Fast CPU mode:

```powershell
.\venv\Scripts\python.exe human_detector.py --width 640 --height 480 --inference-size 320 --camera-fps 30
```

## Controls

- Press `q` in the camera window to quit.

## Useful Options

Try a different webcam:

```powershell
.\venv\Scripts\python.exe human_detector.py --camera 1
```

Use a stricter prediction threshold:

```powershell
.\venv\Scripts\python.exe human_detector.py --confidence 0.70
```

Use a faster, smaller camera size:

```powershell
.\venv\Scripts\python.exe human_detector.py --width 640 --height 480
```

Use an even faster inference size if your FPS is still low:

```powershell
.\venv\Scripts\python.exe human_detector.py --width 640 --height 480 --inference-size 256
```
