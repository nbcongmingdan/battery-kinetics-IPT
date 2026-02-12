import cv2
import numpy as np
import csv
import argparse
from collections import deque

def pca_orientation_from_contour(cnt: np.ndarray) -> float:
    """
    Return orientation angle (radians) of the contour's major axis in image plane.
    Angle in [-pi/2, pi/2) approximately (principal axis direction, 180° ambiguity).
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    X = pts - mean
    # covariance
    C = (X.T @ X) / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    v = eigvecs[:, np.argmax(eigvals)]    # major axis eigenvector
    ang = np.arctan2(v[1], v[0])          # [-pi, pi]
    # fold to [-pi/2, pi/2) because axis has 180° ambiguity
    if ang >= np.pi/2:
        ang -= np.pi
    if ang < -np.pi/2:
        ang += np.pi
    return float(ang)

def unwrap_angles(angle_list):
    """Unwrap a list/array of angles (rad) to be continuous over time."""
    if len(angle_list) == 0:
        return []
    unwrapped = [angle_list[0]]
    for a in angle_list[1:]:
        prev = unwrapped[-1]
        da = a - (prev % (2*np.pi))
        # bring difference to [-pi, pi]
        da = (da + np.pi) % (2*np.pi) - np.pi
        unwrapped.append(prev + da)
    return unwrapped

def finite_diff_omega(theta, dt):
    """Central difference angular velocity; endpoints use forward/backward."""
    n = len(theta)
    omega = [0.0]*n
    if n < 2:
        return omega
    omega[0] = (theta[1] - theta[0]) / dt
    for i in range(1, n-1):
        omega[i] = (theta[i+1] - theta[i-1]) / (2*dt)
    omega[-1] = (theta[-1] - theta[-2]) / dt
    return omega

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input .mp4 path")
    ap.add_argument("--csv", default="angle_omega.csv", help="output csv path")
    ap.add_argument("--debug_video", default="", help="optional debug output mp4")
    ap.add_argument("--show", action="store_true", help="show debug window")
    ap.add_argument("--min_area", type=float, default=80.0, help="min contour area to accept")
    ap.add_argument("--history", type=int, default=200, help="MOG2 history")
    ap.add_argument("--varThreshold", type=float, default=25.0, help="MOG2 varThreshold")
    ap.add_argument("--warmup", type=int, default=20, help="frames to warm up bg subtractor")
    ap.add_argument("--smooth", type=int, default=5, help="moving average window for angle (odd recommended)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Background subtraction for robust segmentation
    bg = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.varThreshold,
        detectShadows=False
    )

    writer = None
    if args.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.debug_video, fourcc, fps, (w, h))

    angles = []
    times = []
    frames_idx = []

    # optional: keep last valid contour to handle dropouts
    last_cnt = None
    last_angle = None

    # for smoothing (angle before unwrap is tricky; we smooth after unwrap instead)
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # light denoise (helps low-light sensor noise)
        gray_dn = cv2.GaussianBlur(gray, (5, 5), 0)

        # foreground mask
        fg = bg.apply(gray_dn)

        # warmup: ignore early frames to let background model settle
        if frame_i < args.warmup:
            frame_i += 1
            if writer:
                writer.write(frame)
            continue

        # clean up mask
        _, bw = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.medianBlur(bw, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        # find contours
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = None
        if cnts:
            # choose largest contour
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < args.min_area:
                cnt = None

        if cnt is None:
            # fallback: reuse last contour/angle if available
            cnt = last_cnt

        if cnt is not None:
            ang = pca_orientation_from_contour(cnt)
            last_cnt = cnt
            last_angle = ang
        else:
            # if everything fails, hold last angle; else skip
            if last_angle is None:
                frame_i += 1
                continue
            ang = last_angle

        t = frame_i * dt
        angles.append(ang)
        times.append(t)
        frames_idx.append(frame_i)

        # debug overlay
        if args.show or writer:
            vis = frame.copy()

            if last_cnt is not None:
                cv2.drawContours(vis, [last_cnt], -1, (0, 255, 0), 2)

                # draw principal axis direction
                M = cv2.moments(last_cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    L = 80
                    dx = int(L * np.cos(ang))
                    dy = int(L * np.sin(ang))
                    cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)
                    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

            cv2.putText(vis, f"t={t:.3f}s  theta={ang:.3f} rad",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if writer:
                writer.write(vis)
            if args.show:
                cv2.imshow("debug", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        frame_i += 1

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # unwrap angles to continuous theta(t)
    theta_unwrapped = unwrap_angles(angles)

    # optional smoothing AFTER unwrap (simple moving average)
    if args.smooth and args.smooth > 1:
        k = int(args.smooth)
        if k % 2 == 0:
            k += 1
        half = k // 2
        padded = np.pad(np.array(theta_unwrapped, dtype=float), (half, half), mode="edge")
        sm = np.convolve(padded, np.ones(k)/k, mode="valid")
        theta_used = sm.tolist()
    else:
        theta_used = theta_unwrapped

    omega = finite_diff_omega(theta_used, dt)

    # write CSV
    with open(args.csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "t_s", "theta_rad_unwrapped", "omega_rad_s"])
        for fr, t, th, om in zip(frames_idx, times, theta_used, omega):
            wcsv.writerow([fr, f"{t:.6f}", f"{th:.9f}", f"{om:.9f}"])

    print(f"Done. FPS={fps:.3f}, samples={len(theta_used)}")
    print(f"Saved: {args.csv}")
    if args.debug_video:
        print(f"Saved debug video: {args.debug_video}")

if __name__ == "__main__":
    main()  