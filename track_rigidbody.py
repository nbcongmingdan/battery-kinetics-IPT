import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from pathlib import Path

VIDEO_PATH = "/home/xuwentao/IPT-2026/test-videos/cut-05.mov"
PIXEL_TO_METER = 0.001
FPS_OVERRIDE = None

SHOW_DEBUG = True
WRITE_DEBUG_VIDEO = True

# --------------------------------------
# 1. 打开视频
# --------------------------------------

video_path = Path(VIDEO_PATH)
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
if FPS_OVERRIDE:
    fps = FPS_OVERRIDE

dt = 1 / fps

writer = None
if WRITE_DEBUG_VIDEO:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    debug_dir = Path("debug_output")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_output_path = debug_dir / f"{video_path.stem}-debug.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(debug_output_path), fourcc, fps, (width, height))

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=20,
    detectShadows=False
)

centers = []
angles = []

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --------------------------------------
# 2. 主循环
# --------------------------------------

for _ in tqdm(range(total_frames)):

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cx, cy, theta = np.nan, np.nan, np.nan

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 200 and len(c) >= 5:

            ellipse = cv2.fitEllipse(c)

            (cx, cy), (MA, ma), angle = ellipse

            theta = np.deg2rad(angle)

            if SHOW_DEBUG:
                cv2.ellipse(frame, ellipse, (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

    centers.append((cx, cy))
    angles.append(theta)

    if SHOW_DEBUG:
        cv2.imshow("Rigid Body Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if writer is not None:
        writer.write(frame)

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

centers = np.array(centers)
angles = np.array(angles)

# --------------------------------------
# 3. 转物理单位
# --------------------------------------

x = centers[:,0] * PIXEL_TO_METER
y = centers[:,1] * PIXEL_TO_METER
theta = angles

time = np.arange(len(x)) * dt

valid = ~np.isnan(x)
x = x[valid]
y = y[valid]
theta = theta[valid]
time = time[valid]

# --------------------------------------
# 4. 角度展开（避免跳变）
# --------------------------------------

theta = np.unwrap(theta)

# --------------------------------------
# 5. Savitzky-Golay 平滑
# --------------------------------------

window = 21 if len(x) > 21 else len(x)//2*2+1

x_s = savgol_filter(x, window, 3)
y_s = savgol_filter(y, window, 3)
theta_s = savgol_filter(theta, window, 3)

vx = np.gradient(x_s, dt)
vy = np.gradient(y_s, dt)
omega = np.gradient(theta_s, dt)

ax = np.gradient(vx, dt)
ay = np.gradient(vy, dt)
alpha = np.gradient(omega, dt)

# --------------------------------------
# 6. 保存
# --------------------------------------

np.savetxt(
    "rigidbody_data.txt",
    np.column_stack([
        time,
        x_s, y_s,
        vx, vy,
        ax, ay,
        theta_s,
        omega,
        alpha
    ]),
    header="t x y vx vy ax ay theta omega alpha"
)

print("Saved rigidbody_data.txt")

# --------------------------------------
# 7. 可视化
# --------------------------------------

plt.figure()
plt.plot(time, y_s)
plt.title("Vertical position")
plt.show()

plt.figure()
plt.plot(time, omega)
plt.title("Angular velocity")
plt.show()

plt.figure()
plt.plot(time, alpha)
plt.title("Angular acceleration")
plt.show()
