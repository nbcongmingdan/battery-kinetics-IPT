import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

VIDEO_PATH = "/home/xuwentao/IPT-2026/test-videos/cut-05.mov"
PIXEL_TO_METER = 0.001
FPS_OVERRIDE = None

SHOW_DEBUG = True          # 👈 是否显示检测框
SAVE_DEBUG_VIDEO = False   # 👈 是否保存带框视频

# --------------------------------------
# 1. 打开视频
# --------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
if FPS_OVERRIDE:
    fps = FPS_OVERRIDE

dt = 1 / fps

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if SAVE_DEBUG_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("debug_output.mp4", fourcc, fps, (width, height))

# --------------------------------------
# 2. 背景模型
# --------------------------------------

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)

centers = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --------------------------------------
# 3. 主循环
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

    cx, cy = np.nan, np.nan

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 50:
            M = cv2.moments(c)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            if SHOW_DEBUG:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)

    centers.append((cx, cy))

    if SHOW_DEBUG:
        cv2.imshow("Detection Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if SAVE_DEBUG_VIDEO:
        out.write(frame)

cap.release()

if SAVE_DEBUG_VIDEO:
    out.release()

if SHOW_DEBUG:
    cv2.destroyAllWindows()

centers = np.array(centers)

# --------------------------------------
# 4. 物理转换
# --------------------------------------

x = centers[:,0] * PIXEL_TO_METER
y = centers[:,1] * PIXEL_TO_METER
time = np.arange(len(x)) * dt

valid = ~np.isnan(x)
x = x[valid]
y = y[valid]
time = time[valid]

# --------------------------------------
# 5. 平滑
# --------------------------------------

window = 21 if len(x) > 21 else len(x)//2*2+1
x_smooth = savgol_filter(x, window, 3)
y_smooth = savgol_filter(y, window, 3)

vx = np.gradient(x_smooth, dt)
vy = np.gradient(y_smooth, dt)
speed = np.sqrt(vx**2 + vy**2)

np.savetxt(
    "trajectory.txt",
    np.column_stack([time, x_smooth, y_smooth, vx, vy, speed]),
    header="t x y vx vy speed"
)

print("Saved trajectory.txt")

# --------------------------------------
# 6. 画图
# --------------------------------------

plt.figure()
plt.plot(time, y_smooth)
plt.title("Vertical position")
plt.xlabel("Time (s)")
plt.ylabel("y (m)")
plt.show()

plt.figure()
plt.plot(time, speed)
plt.title("Speed")
plt.xlabel("Time (s)")
plt.ylabel("m/s")
plt.show()
