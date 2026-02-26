import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from pathlib import Path

VIDEO_PATH = "/home/xuwentao/IPT-2026/test-videos/nu.00/C0007.MP4_ppt-00.00.01.569-00.00.03.187-seg01.mp4"
PIXEL_TO_METER = 0.000364
FPS_OVERRIDE = None

# Uniform solid cylinder parameters
MASS_KG = 0.011
LENGTH_M = 0.044
DIAMETER_M = 0.0102
RADIUS_M = DIAMETER_M / 2.0

# Rotation is in-plane (y-z), so omega is about x-axis (out of plane).
# Use moment of inertia about an axis through center, perpendicular to cylinder axis.
INERTIA_KG_M2 = (1.0 / 12.0) * MASS_KG * (3.0 * RADIUS_M**2 + LENGTH_M**2)
GRAVITY_M_S2 = 9.80665

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
areas = []

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
    area = np.nan

    if contours:
        c = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(c)
        if contour_area > 200 and len(c) >= 5:

            ellipse = cv2.fitEllipse(c)

            (cx, cy), (MA, ma), angle = ellipse

            theta = np.deg2rad(angle)
            area = contour_area

            if SHOW_DEBUG:
                cv2.ellipse(frame, ellipse, (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

    centers.append((cx, cy))
    angles.append(theta)
    areas.append(area)

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
areas = np.array(areas)

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
area = areas[valid]

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
area_s = savgol_filter(area, window, 3)

vx = savgol_filter(x, window, 3, deriv=1, delta=dt)
vy = savgol_filter(y, window, 3, deriv=1, delta=dt)
omega = savgol_filter(theta, window, 3, deriv=1, delta=dt)

ax = savgol_filter(x, window, 3, deriv=2, delta=dt)
ay = savgol_filter(y, window, 3, deriv=2, delta=dt)
alpha = savgol_filter(theta, window, 3, deriv=2, delta=dt)

# --------------------------------------
# 6. 计算绕 y 轴角度（由 yz 投影面积）
# --------------------------------------

area_m2 = area_s * (PIXEL_TO_METER ** 2)

proj_a = 2.0 * RADIUS_M * LENGTH_M
proj_b = np.pi * RADIUS_M ** 2
proj_r = np.sqrt(proj_a ** 2 + proj_b ** 2)
proj_delta = np.arctan2(proj_b, proj_a)

arg = np.clip(area_m2 / proj_r, -1.0, 1.0)
acos_val = np.arccos(arg)
phi1 = proj_delta + acos_val
phi2 = proj_delta - acos_val
phi = np.where((phi1 >= 0.0) & (phi1 <= np.pi / 2.0), phi1, phi2)
phi = np.clip(phi, 0.0, np.pi / 2.0)

omega_y = savgol_filter(phi, window, 3, deriv=1, delta=dt)
alpha_y = savgol_filter(phi, window, 3, deriv=2, delta=dt)

# --------------------------------------
# 7. 动能计算
# --------------------------------------

speed = np.sqrt(vx**2 + vy**2)
rot_ke = 0.5 * INERTIA_KG_M2 * (omega**2 + omega_y**2)
kinetic_energy = 0.5 * MASS_KG * speed**2 + rot_ke
potential_energy = MASS_KG * GRAVITY_M_S2 * (y_s[0] - y_s)
total_energy = kinetic_energy + potential_energy

# --------------------------------------
# 8. 保存
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
        alpha,
        phi,
        omega_y,
        alpha_y,
        kinetic_energy,
        potential_energy,
        total_energy
    ]),
    header="t x y vx vy ax ay theta omega alpha phi omega_y alpha_y kinetic_energy potential_energy total_energy"
)

print("Saved rigidbody_data.txt")

# --------------------------------------
# 9. 可视化
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

plt.figure()
plt.plot(time, kinetic_energy)
plt.title("Kinetic energy")
plt.xlabel("Time (s)")
plt.ylabel("J")
plt.show()

plt.figure()
plt.plot(time, total_energy)
plt.title("Total energy")
plt.xlabel("Time (s)")
plt.ylabel("J")
plt.tight_layout()
plt.savefig("total_energy.png", dpi=200)
plt.show()
