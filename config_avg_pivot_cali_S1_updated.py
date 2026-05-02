import cv2
import json
import numpy as np
import os


# ============================================================
# Globals for mouse clicks
# ============================================================

clicked_point = None


def mouse_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


# ============================================================
# Utility helpers
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_hsv_range_from_patch(hsv_patch, hue_pad=36, sat_pad=75, val_pad=75):
    h = hsv_patch[:, :, 0].reshape(-1)
    s = hsv_patch[:, :, 1].reshape(-1)
    v = hsv_patch[:, :, 2].reshape(-1)

    h_med = int(np.median(h))
    s_med = int(np.median(s))
    v_med = int(np.median(v))

    h_lo = h_med - hue_pad
    h_hi = h_med + hue_pad
    s_lo = clamp(s_med - sat_pad, 0, 255)
    s_hi = clamp(s_med + sat_pad, 0, 255)
    v_lo = clamp(v_med - val_pad, 0, 255)
    v_hi = clamp(v_med + val_pad, 0, 255)

    ranges = []

    if h_lo < 0:
        ranges.append({"lower": [0, s_lo, v_lo], "upper": [h_hi, s_hi, v_hi]})
        ranges.append({"lower": [180 + h_lo, s_lo, v_lo], "upper": [179, s_hi, v_hi]})
    elif h_hi > 179:
        ranges.append({"lower": [0, s_lo, v_lo], "upper": [h_hi - 180, s_hi, v_hi]})
        ranges.append({"lower": [h_lo, s_lo, v_lo], "upper": [179, s_hi, v_hi]})
    else:
        ranges.append({"lower": [h_lo, s_lo, v_lo], "upper": [h_hi, s_hi, v_hi]})

    return {
        "median_hsv": [h_med, s_med, v_med],
        "ranges": ranges
    }


def apply_hsv_ranges(hsv_img, hsv_range_dict):
    mask_total = None
    for r in hsv_range_dict["ranges"]:
        lower = np.array(r["lower"], dtype=np.uint8)
        upper = np.array(r["upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
        if mask_total is None:
            mask_total = mask
        else:
            mask_total = cv2.bitwise_or(mask_total, mask)
    return mask_total


def sample_patch(hsv_img, center_xy, half_size=6):
    x, y = center_xy
    h, w = hsv_img.shape[:2]
    x0 = max(0, x - half_size)
    x1 = min(w, x + half_size + 1)
    y0 = max(0, y - half_size)
    y1 = min(h, y + half_size + 1)
    return hsv_img[y0:y1, x0:x1], (x0, y0, x1, y1)


def clean_mask(mask, open_kernel=2, close_kernel=5, dilate_kernel=0):
    mask_out = mask.copy()

    if open_kernel and open_kernel > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, k_open)

    if close_kernel and close_kernel > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, k_close)

    if dilate_kernel and dilate_kernel > 1:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask_out = cv2.dilate(mask_out, k_dil, iterations=1)

    return mask_out


def preview_mask_on_frame(frame_bgr, mask):
    overlay = frame_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def largest_contour(mask, min_area=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def contour_centroid(contour):
    if contour is None or len(contour) < 3:
        return None
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-9:
        return None
    return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))


def contour_diameter_metrics(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    d_eq = np.sqrt(4.0 * area / np.pi)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    d_circ = 2.0 * radius

    return {
        "area_px2": float(area),
        "equivalent_diameter_px": float(d_eq),
        "enclosing_circle_diameter_px": float(d_circ)
    }


def read_roi_frame_at(video_path, roi_xywh, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    x, y, w, h = roi_xywh
    roi_frame = frame[y:y+h, x:x+w].copy()
    return roi_frame


def detect_marker_on_roi_frame(roi_frame_bgr, hsv_cfg, tracking_defaults):
    hsv_roi = cv2.cvtColor(roi_frame_bgr, cv2.COLOR_BGR2HSV)
    mask_raw = apply_hsv_ranges(hsv_roi, hsv_cfg)
    mask = clean_mask(
        mask_raw,
        open_kernel=int(tracking_defaults.get("morph_open_kernel", 3)),
        close_kernel=int(tracking_defaults.get("morph_close_kernel", 5))
    )

    contour = largest_contour(mask, min_area=int(tracking_defaults.get("min_contour_area_px", 30)))
    ctr = contour_centroid(contour)
    metrics = contour_diameter_metrics(contour) if contour is not None else None

    return ctr, metrics, mask, contour


def get_three_windows(frame_count, window_size=10):
    start_indices = list(range(0, min(window_size, frame_count)))

    mid_start = max(0, min(frame_count - window_size, frame_count // 2 - window_size // 2))
    mid_indices = list(range(mid_start, min(mid_start + window_size, frame_count)))

    end_start = max(0, frame_count - window_size)
    end_indices = list(range(end_start, frame_count))

    return {
        "start": start_indices,
        "middle": mid_indices,
        "end": end_indices
    }


def summarize_values(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        return {
            "n_valid": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }

    arr = np.array(vals, dtype=float)
    return {
        "n_valid": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def collect_marker_diameter_stats(video_path, roi_xywh, frame_indices, hsv_cfg, tracking_defaults):
    eq_diams = []
    circ_diams = []
    areas = []
    per_frame = []

    for idx in frame_indices:
        roi_frame = read_roi_frame_at(video_path, roi_xywh, idx)
        if roi_frame is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        ctr, metrics, mask, contour = detect_marker_on_roi_frame(roi_frame, hsv_cfg, tracking_defaults)

        if metrics is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        eq_diams.append(metrics["equivalent_diameter_px"])
        circ_diams.append(metrics["enclosing_circle_diameter_px"])
        areas.append(metrics["area_px2"])

        per_frame.append({
            "frame_idx": int(idx),
            "valid": True,
            "equivalent_diameter_px": float(metrics["equivalent_diameter_px"]),
            "enclosing_circle_diameter_px": float(metrics["enclosing_circle_diameter_px"]),
            "area_px2": float(metrics["area_px2"])
        })

    return {
        "equivalent_diameters_px": eq_diams,
        "enclosing_circle_diameters_px": circ_diams,
        "areas_px2": areas,
        "summary_equivalent_diameter_px": summarize_values(eq_diams),
        "summary_enclosing_circle_diameter_px": summarize_values(circ_diams),
        "summary_area_px2": summarize_values(areas),
        "per_frame": per_frame
    }


def collect_marker_x_stats(video_path, roi_xywh, frame_indices, hsv_cfg, tracking_defaults):
    xs = []
    per_frame = []

    for idx in frame_indices:
        roi_frame = read_roi_frame_at(video_path, roi_xywh, idx)
        if roi_frame is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        ctr, metrics, mask, contour = detect_marker_on_roi_frame(roi_frame, hsv_cfg, tracking_defaults)

        if ctr is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        xs.append(float(ctr[0]))
        per_frame.append({
            "frame_idx": int(idx),
            "valid": True,
            "x_px": float(ctr[0]),
            "y_px": float(ctr[1])
        })

    return {
        "x_values_px": xs,
        "summary_x_px": summarize_values(xs),
        "per_frame": per_frame
    }


def click_point_on_image(window_name, image_bgr, prompt_text,
                         prompt_color=(0, 255, 255),
                         max_display_width=900,
                         max_display_height=600):
    global clicked_point
    clicked_point = None

    h, w = image_bgr.shape[:2]

    scale_w = max_display_width / float(w)
    scale_h = max_display_height / float(h)
    display_scale = min(scale_w, scale_h, 1.0)   # never enlarge, only shrink if needed

    if display_scale < 1.0:
        disp_img = cv2.resize(
            image_bgr,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        disp_img = image_bgr.copy()

    base = disp_img.copy()
    cv2.putText(base, prompt_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, prompt_color, 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, base)
    cv2.resizeWindow(window_name, base.shape[1], base.shape[0])
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        disp = base.copy()

        if clicked_point is not None:
            cv2.circle(disp, clicked_point, 6, prompt_color, 2)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if clicked_point is not None:
            pt_disp = clicked_point

            # map display click back to original image coordinates
            x_orig = int(round(pt_disp[0] / display_scale))
            y_orig = int(round(pt_disp[1] / display_scale))

            x_orig = max(0, min(w - 1, x_orig))
            y_orig = max(0, min(h - 1, y_orig))

            cv2.destroyWindow(window_name)
            return (x_orig, y_orig)

        if key == 27:
            cv2.destroyAllWindows()
            return None


def format_signed_sum(base, offset):
    """
    Format base + offset = result cleanly.
    Example:
      228.282 + 4.718 = 233.000
      478.740 - 9.740 = 469.000
    """
    base = float(base)
    offset = float(offset)
    result = base + offset

    sign = "+" if offset >= 0 else "-"
    return f"{base:.3f} {sign} {abs(offset):.3f} = {result:.3f}"


def click_true_pivot_confirmable(
    window_name,
    image_bgr,
    green_ctr_xy,
    prompt_text,
    max_display_width=900,
    max_display_height=600,
    warning_offset_px=80.0
):
    """
    Re-clickable true-pivot picker.

    Controls:
      - Left click: place/re-place true pivot point
      - Enter or Space: accept current point
      - R: reset current point
      - ESC: cancel

    Shows:
      - green marker center in green
      - true pivot candidate in red
      - line from green marker to true pivot
      - live x_green + dx = true_x calculation
    """
    global clicked_point
    clicked_point = None

    h, w = image_bgr.shape[:2]

    scale_w = max_display_width / float(w)
    scale_h = max_display_height / float(h)
    display_scale = min(scale_w, scale_h, 1.0)

    if display_scale < 1.0:
        base_display = cv2.resize(
            image_bgr,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        base_display = image_bgr.copy()

    green_x = float(green_ctr_xy[0])
    green_y = float(green_ctr_xy[1])

    green_disp = (
        int(round(green_x * display_scale)),
        int(round(green_y * display_scale))
    )

    candidate_orig = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback(window_name, mouse_click)

    print("\nTrue pivot picker controls:")
    print("  Left click = place/re-place red true pivot")
    print("  Enter/Space = accept")
    print("  R = reset")
    print("  ESC = cancel")

    while True:
        disp = base_display.copy()

        # Draw detected green marker center
        cv2.circle(disp, green_disp, 7, (0, 255, 0), 2)
        cv2.circle(disp, green_disp, 2, (0, 255, 0), -1)

        cv2.putText(
            disp,
            prompt_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            disp,
            "Left click/re-click true pivot. Enter/Space=accept, R=reset, ESC=cancel.",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if candidate_orig is not None:
            true_x = float(candidate_orig[0])
            true_y = float(candidate_orig[1])
            dx = true_x - green_x
            dy = true_y - green_y

            candidate_disp = (
                int(round(true_x * display_scale)),
                int(round(true_y * display_scale))
            )

            # Red true-pivot candidate
            cv2.circle(disp, candidate_disp, 8, (0, 0, 255), 2)
            cv2.circle(disp, candidate_disp, 3, (0, 0, 255), -1)
            cv2.line(disp, green_disp, candidate_disp, (255, 255, 255), 2)

            x_text = "x: " + format_signed_sum(green_x, dx) + " px"
            y_text = "y: " + format_signed_sum(green_y, dy) + " px"
            offset_text = f"dx={dx:+.3f}px, dy={dy:+.3f}px"

            cv2.putText(
                disp,
                x_text,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )
            cv2.putText(
                disp,
                y_text,
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )
            cv2.putText(
                disp,
                offset_text,
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            if abs(dx) > warning_offset_px or abs(dy) > warning_offset_px:
                cv2.putText(
                    disp,
                    "WARNING: offset is very large. Re-click unless this is intentional.",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF

        # Process new click, if any
        if clicked_point is not None:
            x_disp, y_disp = clicked_point

            x_orig = int(round(x_disp / display_scale))
            y_orig = int(round(y_disp / display_scale))

            x_orig = max(0, min(w - 1, x_orig))
            y_orig = max(0, min(h - 1, y_orig))

            candidate_orig = (x_orig, y_orig)
            clicked_point = None

            dx = float(candidate_orig[0]) - green_x
            dy = float(candidate_orig[1]) - green_y

            print("\nCurrent true pivot candidate:")
            print(f"  x: {format_signed_sum(green_x, dx)} px")
            print(f"  y: {format_signed_sum(green_y, dy)} px")
            print(f"  dx={dx:+.3f} px, dy={dy:+.3f} px")

            if abs(dx) > warning_offset_px or abs(dy) > warning_offset_px:
                print("  WARNING: offset is very large. Re-click unless this is intentional.")

        # Enter or Space = accept
        if key in (13, 10, 32):
            if candidate_orig is not None:
                cv2.destroyWindow(window_name)
                return candidate_orig

        # R = reset
        if key in (ord("r"), ord("R")):
            candidate_orig = None
            print("  Reset current true pivot candidate. Click again.")

        # ESC = cancel
        if key == 27:
            cv2.destroyAllWindows()
            return None

def click_marker_center_confirmable(
    window_name,
    image_bgr,
    detected_ctr_xy,
    prompt_text,
    marker_color_bgr=(0, 255, 0),
    max_display_width=900,
    max_display_height=600
):
    """
    Re-clickable marker-center picker.

    Controls:
      - Left click: place/re-place marker center
      - Enter or Space: accept current point
      - R: reset back to detected point
      - ESC: cancel

    Use this when automatic HSV detection may grab the wrong object.
    """
    global clicked_point
    clicked_point = None

    h, w = image_bgr.shape[:2]

    scale_w = max_display_width / float(w)
    scale_h = max_display_height / float(h)
    display_scale = min(scale_w, scale_h, 1.0)

    if display_scale < 1.0:
        base_display = cv2.resize(
            image_bgr,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        base_display = image_bgr.copy()

    if detected_ctr_xy is not None:
        candidate_orig = (
            int(round(detected_ctr_xy[0])),
            int(round(detected_ctr_xy[1]))
        )
    else:
        candidate_orig = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback(window_name, mouse_click)

    print("\nMarker center confirmation controls:")
    print("  Left click = place/re-place marker center")
    print("  Enter/Space = accept")
    print("  R = reset to detected center")
    print("  ESC = cancel")

    while True:
        disp = base_display.copy()

        cv2.putText(
            disp,
            prompt_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            disp,
            "Confirm marker center. Left click/re-click, Enter/Space=accept, R=reset, ESC=cancel.",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if detected_ctr_xy is not None:
            detected_disp = (
                int(round(float(detected_ctr_xy[0]) * display_scale)),
                int(round(float(detected_ctr_xy[1]) * display_scale))
            )
            cv2.circle(disp, detected_disp, 10, (0, 255, 255), 2)
            cv2.putText(
                disp,
                "auto",
                (detected_disp[0] + 8, detected_disp[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        if candidate_orig is not None:
            candidate_disp = (
                int(round(candidate_orig[0] * display_scale)),
                int(round(candidate_orig[1] * display_scale))
            )
            cv2.circle(disp, candidate_disp, 8, marker_color_bgr, 2)
            cv2.circle(disp, candidate_disp, 3, marker_color_bgr, -1)

            cv2.putText(
                disp,
                f"selected center: x={candidate_orig[0]:.3f}, y={candidate_orig[1]:.3f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                marker_color_bgr,
                2
            )

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if clicked_point is not None:
            x_disp, y_disp = clicked_point

            x_orig = int(round(x_disp / display_scale))
            y_orig = int(round(y_disp / display_scale))

            x_orig = max(0, min(w - 1, x_orig))
            y_orig = max(0, min(h - 1, y_orig))

            candidate_orig = (x_orig, y_orig)
            clicked_point = None

            print(f"  Current marker center candidate: x={x_orig:.3f}, y={y_orig:.3f}")

        if key in (13, 10, 32):
            if candidate_orig is not None:
                cv2.destroyWindow(window_name)
                return candidate_orig

        if key in (ord("r"), ord("R")):
            if detected_ctr_xy is not None:
                candidate_orig = (
                    int(round(detected_ctr_xy[0])),
                    int(round(detected_ctr_xy[1]))
                )
                print("  Reset to auto-detected marker center.")
            else:
                candidate_orig = None
                print("  Reset marker center. Click manually.")

        if key == 27:
            cv2.destroyAllWindows()
            return None

def print_pivot_sample_summary(offset_samples):
    """
    Print a readable summary of x_green + dx = x_true for each pivot sample.
    Also flags suspicious jumps.
    """
    if not offset_samples:
        print("\nNo pivot offset samples to summarize.")
        return

    rows = []
    for s in offset_samples:
        xg = float(s["green_marker_center_roi_px"][0])
        yg = float(s["green_marker_center_roi_px"][1])
        dx = float(s["pivot_offset_x_px"])
        dy = float(s["pivot_offset_y_px"])
        xt = xg + dx
        yt = yg + dy
        rows.append((xg, yg, dx, dy, xt, yt, s))

    rows.sort(key=lambda r: r[0])

    print("\nPivot sample summary:")
    print("  x equation:")
    for xg, yg, dx, dy, xt, yt, s in rows:
        print(
            f"  {s['label']:<10} frame {s['frame_idx']:>6}: "
            f"x: {format_signed_sum(xg, dx)} px, "
            f"y: {format_signed_sum(yg, dy)} px"
        )

    dx_vals = np.array([r[2] for r in rows], dtype=float)
    dy_vals = np.array([r[3] for r in rows], dtype=float)
    xt_vals = np.array([r[4] for r in rows], dtype=float)

    print("\nPivot sample spread:")
    print(f"  dx min/max/range [px] = {np.min(dx_vals):+.3f}, {np.max(dx_vals):+.3f}, {np.ptp(dx_vals):.3f}")
    print(f"  dy min/max/range [px] = {np.min(dy_vals):+.3f}, {np.max(dy_vals):+.3f}, {np.ptp(dy_vals):.3f}")

    if len(xt_vals) >= 2:
        true_x_jumps = np.diff(xt_vals)
        print(f"  true pivot x jumps between sorted samples [px]: {true_x_jumps}")

        large_jump_idx = np.where(np.abs(true_x_jumps) > 150.0)[0]
        if large_jump_idx.size > 0:
            print("  WARNING: large jump(s) in true pivot x detected.")
            print("  This usually means one sample was clicked/detected wrong.")

    bad_offset_idx = np.where((np.abs(dx_vals) > 80.0) | (np.abs(dy_vals) > 80.0))[0]
    if bad_offset_idx.size > 0:
        print("  WARNING: one or more pivot offsets exceed 80 px.")
        print("  Check/re-click those samples.")


def print_window_summary(title, stats):
    s = stats["summary_equivalent_diameter_px"]
    if s["n_valid"] == 0:
        print(f"  {title:<8}: no valid detections")
    else:
        print(
            f"  {title:<8}: n={s['n_valid']}, "
            f"mean={s['mean']:.3f}, median={s['median']:.3f}, std={s['std']:.3f} px/in"
        )

def select_pivot_offset_samples(offset_samples, selection_mode="all", manual_labels=None):
    """
    Choose which pivot offset samples to use and return the active average.
    selection_mode:
        - "all"
        - "middle_end"
        - "manual"
    manual_labels:
        list like ["middle", "end"]
    """
    if not offset_samples:
        raise ValueError("No pivot offset samples provided.")

    if selection_mode == "all":
        selected = list(offset_samples)

    elif selection_mode == "middle_end":
        selected = [s for s in offset_samples if s["label"] in ("middle", "end")]
        if len(selected) == 0:
            selected = list(offset_samples)

    elif selection_mode == "manual":
        manual_labels = manual_labels or []
        selected = [s for s in offset_samples if s["label"] in manual_labels]
        if len(selected) == 0:
            raise ValueError("Manual pivot selection produced no valid samples.")

    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    dx_vals = [float(s["pivot_offset_x_px"]) for s in selected]
    dy_vals = [float(s["pivot_offset_y_px"]) for s in selected]

    return {
        "selected_samples": selected,
        "pivot_offset_x_px_active": float(np.mean(dx_vals)),
        "pivot_offset_y_px_active": float(np.mean(dy_vals))
    }

def get_biased_pivot_sample_indices(
    frame_count,
    fps_used,
    total_samples=10,
    early_time_s=5.0,
    early_samples=5,
    start_margin_frames=200,
    end_margin_frames=200
):
    """
    Build sample indices with denser coverage during the first early_time_s,
    then spread the remaining samples across the rest of the video.
    """
    frame_count = int(frame_count)
    total_samples = int(total_samples)

    if frame_count <= 0 or total_samples < 2:
        return []

    early_end = int(round(early_time_s * fps_used))
    early_end = max(0, min(frame_count - 1, early_end))

    start_idx = max(0, int(start_margin_frames))
    final_idx = max(start_idx, frame_count - 1 - int(end_margin_frames))

    early_samples = max(1, min(total_samples - 1, int(early_samples)))
    late_samples = total_samples - early_samples

    indices = []

    # early dense region
    if early_end > start_idx:
        early_idx = np.linspace(start_idx, early_end, early_samples)
        indices.extend(int(round(v)) for v in early_idx)
    else:
        indices.append(start_idx)

    # later spread region
    if late_samples > 0 and final_idx > early_end:
        late_start = min(final_idx, max(early_end + 1, early_end + 200))
        if final_idx > late_start:
            late_idx = np.linspace(late_start, final_idx, late_samples)
            indices.extend(int(round(v)) for v in late_idx)
        else:
            indices.extend([final_idx] * late_samples)

    # unique + sorted
    indices = sorted(set(indices))

    return indices


def fit_piecewise_linear_pivot_model(offset_samples):
    """
    Build a piecewise-linear pivot offset model as a function of green-marker x-position.
    Saves nodes for later np.interp use in analysis.
    """
    if len(offset_samples) < 2:
        raise ValueError("Need at least 2 offset samples for an interpolated pivot model.")

    rows = []
    for s in offset_samples:
        xg = float(s["green_marker_center_roi_px"][0])
        dx = float(s["pivot_offset_x_px"])
        dy = float(s["pivot_offset_y_px"])
        rows.append((xg, dx, dy, s))

    rows.sort(key=lambda r: r[0])

    x_nodes = [r[0] for r in rows]
    dx_nodes = [r[1] for r in rows]
    dy_nodes = [r[2] for r in rows]
    sorted_samples = [r[3] for r in rows]

    return {
        "model_type": "piecewise_linear",
        "x_green_nodes_px": x_nodes,
        "dx_nodes_px": dx_nodes,
        "dy_nodes_px": dy_nodes,
        "x_range_px": [float(min(x_nodes)), float(max(x_nodes))],
        "samples_sorted_by_x": sorted_samples
    }


def contour_rect_metrics(contour):
    """
    For a strip like red tape, minAreaRect is useful.
    The LONG side is used when the strip's known dimension is its length.
    """
    if contour is None or len(contour) < 3:
        return None

    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    width_short = float(min(w, h))
    width_long = float(max(w, h))

    return {
        "area_px2": float(area),
        "width_short_px": width_short,
        "width_long_px": width_long,
        "angle_deg": float(angle)
    }


def collect_tape_length_stats(video_path, roi_xywh, frame_indices, hsv_cfg, tracking_defaults):
    """
    For the red tape, the known reference is 1 inch LONG, not wide.
    So we use the LONG side of minAreaRect as px/in.
    """
    lengths_long = []
    widths_short = []
    areas = []
    per_frame = []

    for idx in frame_indices:
        roi_frame = read_roi_frame_at(video_path, roi_xywh, idx)
        if roi_frame is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask_raw = apply_hsv_ranges(hsv_roi, hsv_cfg)
        mask = clean_mask(
            mask_raw,
            open_kernel=int(tracking_defaults.get("morph_open_kernel", 2)),
            close_kernel=int(tracking_defaults.get("morph_close_kernel", 5)),
            dilate_kernel=int(tracking_defaults.get("dilate_kernel", 0))
        )

        contour = largest_contour(mask, min_area=int(tracking_defaults.get("min_contour_area_px", 30)))
        metrics = contour_rect_metrics(contour) if contour is not None else None

        if metrics is None:
            per_frame.append({"frame_idx": int(idx), "valid": False})
            continue

        lengths_long.append(metrics["width_long_px"])
        widths_short.append(metrics["width_short_px"])
        areas.append(metrics["area_px2"])

        per_frame.append({
            "frame_idx": int(idx),
            "valid": True,
            "length_long_px": float(metrics["width_long_px"]),
            "width_short_px": float(metrics["width_short_px"]),
            "area_px2": float(metrics["area_px2"])
        })

    return {
        "lengths_long_px": lengths_long,
        "widths_short_px": widths_short,
        "areas_px2": areas,
        "summary_length_long_px": summarize_values(lengths_long),
        "summary_width_short_px": summarize_values(widths_short),
        "summary_area_px2": summarize_values(areas),
        "per_frame": per_frame
    }


def read_selected_roi_frames_sequential(video_path, roi_xywh, wanted_indices):
    wanted = sorted(set(int(i) for i in wanted_indices))
    frames = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    x, y, w, h = roi_xywh
    wanted_set = set(wanted)
    k = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if k in wanted_set:
            frames[k] = frame[y:y+h, x:x+w].copy()
            if len(frames) == len(wanted):
                break

        k += 1

    cap.release()
    return frames

def resize_for_display(img, max_display_width=900, max_display_height=600):
    h, w = img.shape[:2]
    scale = min(max_display_width / float(w), max_display_height / float(h), 1.0)
    if scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0

def select_roi_scaled(frame_bgr, window_name="Select ROI",
                      max_display_width=900, max_display_height=600):
    disp, scale = resize_for_display(frame_bgr, max_display_width, max_display_height)
    roi_disp = cv2.selectROI(window_name, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi_disp
    x = int(round(x / scale))
    y = int(round(y / scale))
    w = int(round(w / scale))
    h = int(round(h / scale))
    return (x, y, w, h)
# ============================================================
# Main calibration flow
# ============================================================

def main():
    global clicked_point

    print("\n=== Dual Marker Pendulum Calibration ===\n")

    video_path = input("Enter full path to video file: ").strip().strip('"')

    if not os.path.isfile(video_path):
        print(f"ERROR: File not found:\n{video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo metadata:")
    print(f"  FPS from file: {fps_meta:.3f}")
    print(f"  Frame count  : {frame_count}")

    fps_override_txt = input(
        "\nPress Enter to accept metadata FPS, or type manual FPS (e.g. 240): "
    ).strip()

    fps_used = float(fps_override_txt) if fps_override_txt else float(fps_meta)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("ERROR: Could not read first frame.")
        return

    frame_display = frame.copy()

    print("\nDraw ROI around the experiment and press ENTER or SPACE.")
    roi = select_roi_scaled(frame_display, "Select ROI", 900, 600)
    # Old Select
    #roi = cv2.selectROI("Select ROI", frame_display, showCrosshair=True, fromCenter=False)
    #cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("ERROR: ROI not selected.")
        return

    roi_xywh = [int(x), int(y), int(w), int(h)]
    roi_frame = frame[y:y+h, x:x+w].copy()
    roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    frame_windows = get_three_windows(frame_count, window_size=10)
    representative_frames = {
        "start": frame_windows["start"][len(frame_windows["start"]) // 2],
        "middle": frame_windows["middle"][len(frame_windows["middle"]) // 2],
        "end": frame_windows["end"][len(frame_windows["end"]) // 2]
    }

    # --------------------------------------------------------
    # Click GREEN pivot marker
    # --------------------------------------------------------
    green_click = click_point_on_image(
        "Click GREEN pivot marker",
        roi_frame,
        "Click GREEN pivot marker",
        prompt_color=(0, 255, 0)
    )
    if green_click is None:
        print("Cancelled.")
        return
    green_pt = green_click

    green_patch, green_box = sample_patch(roi_hsv, green_pt, half_size=6)
    
    # --------------------------------------------------------
    # TUNE GREEN pivot marker**********
    # --------------------------------------------------------
    
    green_hsv = build_hsv_range_from_patch(
    green_patch,
    hue_pad=18,
    sat_pad=120,
    val_pad=90
)

    # --------------------------------------------------------
    # Click PINK bob marker
    # --------------------------------------------------------
    pink_click = click_point_on_image(
        "Click PINK bob marker",
        roi_frame,
        "Click PINK bob marker",
        prompt_color=(255, 0, 255)
    )
    if pink_click is None:
        print("Cancelled.")
        return
    pink_pt = pink_click

    pink_patch, pink_box = sample_patch(roi_hsv, pink_pt, half_size=6)
    pink_hsv = build_hsv_range_from_patch(pink_patch)

    tracking_defaults = {
    "min_contour_area_px": 32,
    "morph_open_kernel": 3,
    "morph_close_kernel": 5,
    "dilate_kernel": 4,
    "search_radius_px": 20,
    "max_jump_px": 60
    }

    # --------------------------------------------------------
    # Pivot offset calibration
    # --------------------------------------------------------
    pivot_offset_x_px = 0.0
    pivot_offset_y_px = 0.0
    pivot_offset_info = {}

    print("\nPivot offset calibration:")
    print("  1) Use green marker center as pivot")
    print("  2) Click true physical pivot point on first frame only")
    print("  3) Click true physical pivot point on representative start/middle/end frames and choose active subset")
    print("  4) Click true physical pivot point on 5-7 frames across travel and build x-dependent offset model")
    pivot_choice = input("Choose 1, 2, 3, or 4 [default 1]: ").strip()

    if pivot_choice == "2":
        temp = roi_frame.copy()
        cv2.circle(temp, green_pt, 6, (0, 255, 0), -1)

        true_pivot_pt = click_point_on_image(
            "Click TRUE pivot point",
            temp,
            "Click TRUE pivot point",
            prompt_color=(0, 255, 255)
        )

        if true_pivot_pt is None:
            print("Cancelled.")
            return

        pivot_offset_x_px = float(true_pivot_pt[0] - green_pt[0])
        pivot_offset_y_px = float(true_pivot_pt[1] - green_pt[1])

        print("\nPivot offset result:")
        print(f"  Green marker center : {green_pt}")
        print(f"  True pivot point    : {true_pivot_pt}")
        print(f"  Offset x [px]       : {pivot_offset_x_px:.3f}")
        print(f"  Offset y [px]       : {pivot_offset_y_px:.3f}")

        pivot_offset_info = {
            "mode": "clicked_true_pivot_single_frame",
            "green_marker_center_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "true_pivot_roi_px": [int(true_pivot_pt[0]), int(true_pivot_pt[1])],
            "pivot_offset_x_px_active": pivot_offset_x_px,
            "pivot_offset_y_px_active": pivot_offset_y_px,

            # legacy-compatible fields
            "pivot_offset_x_px": pivot_offset_x_px,
            "pivot_offset_y_px": pivot_offset_y_px
        }

    elif pivot_choice == "3":
        offset_samples = []

        for label, frame_idx_rep in representative_frames.items():
            rep_frame = read_roi_frame_at(video_path, roi_xywh, frame_idx_rep)
            if rep_frame is None:
                print(f"  WARNING: Could not read {label} frame {frame_idx_rep}. Skipping.")
                continue

            green_ctr_rep, green_metrics_rep, green_mask_rep, green_contour_rep = detect_marker_on_roi_frame(
                rep_frame, green_hsv, tracking_defaults
            )

            if green_ctr_rep is None:
                print(f"  WARNING: Could not detect green marker on {label} frame {frame_idx_rep}. Skipping.")
                continue

            temp = rep_frame.copy()
            green_ctr_i = (int(round(green_ctr_rep[0])), int(round(green_ctr_rep[1])))
            cv2.circle(temp, green_ctr_i, 6, (0, 255, 0), -1)
            cv2.putText(
                temp,
                f"{label} frame {frame_idx_rep}: click TRUE pivot",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            true_pivot_pt = click_point_on_image(
                f"TRUE pivot - {label}",
                temp,
                f"{label} frame {frame_idx_rep}: click TRUE pivot",
                prompt_color=(0, 255, 255)
            )

            if true_pivot_pt is None:
                print("Cancelled.")
                return

            dx = float(true_pivot_pt[0] - green_ctr_rep[0])
            dy = float(true_pivot_pt[1] - green_ctr_rep[1])

            offset_samples.append({
                "label": label,
                "frame_idx": int(frame_idx_rep),
                "green_marker_center_roi_px": [float(green_ctr_rep[0]), float(green_ctr_rep[1])],
                "true_pivot_roi_px": [int(true_pivot_pt[0]), int(true_pivot_pt[1])],
                "pivot_offset_x_px": dx,
                "pivot_offset_y_px": dy
            })

        if len(offset_samples) == 0:
            print("ERROR: No valid pivot offset samples were collected.")
            return

        print("\nCollected pivot offset samples:")
        for s in offset_samples:
            print(
                f"  {s['label']:<6} frame {s['frame_idx']:>6}: "
                f"dx={s['pivot_offset_x_px']:+.3f} px, dy={s['pivot_offset_y_px']:+.3f} px"
            )

        dx_vals_all = [float(s["pivot_offset_x_px"]) for s in offset_samples]
        dy_vals_all = [float(s["pivot_offset_y_px"]) for s in offset_samples]
        dx_range = max(dx_vals_all) - min(dx_vals_all)
        dy_range = max(dy_vals_all) - min(dy_vals_all)

        print(f"\nPivot offset spread across sampled frames:")
        print(f"  dx range [px] = {dx_range:.3f}")
        print(f"  dy range [px] = {dy_range:.3f}")

        if dx_range > 10 or dy_range > 10:
            print("  WARNING: Pivot offset samples vary a lot.")
            print("  A single constant pixel offset may be imperfect across the whole travel.")

        print("\nChoose active pivot offset selection:")
        print("  1) Use all valid samples")
        print("  2) Use middle + end only")
        print("  3) Choose labels manually")
        selection_choice = input("Choose 1, 2, or 3 [default 2]: ").strip()

        if selection_choice == "1":
            selection_mode = "all"
            selected_labels = [s["label"] for s in offset_samples]
            selected_result = select_pivot_offset_samples(offset_samples, selection_mode="all")

        elif selection_choice == "3":
            manual_txt = input("Enter labels comma-separated (e.g. middle,end): ").strip().lower()
            manual_labels = [lab.strip() for lab in manual_txt.split(",") if lab.strip()]
            try:
                selected_result = select_pivot_offset_samples(
                    offset_samples,
                    selection_mode="manual",
                    manual_labels=manual_labels
                )
                selection_mode = "manual"
                selected_labels = list(manual_labels)
            except Exception as e:
                print(f"  Manual selection failed: {e}")
                print("  Falling back to middle_end.")
                selected_result = select_pivot_offset_samples(offset_samples, selection_mode="middle_end")
                selection_mode = "middle_end"
                selected_labels = [s["label"] for s in selected_result["selected_samples"]]

        else:
            selected_result = select_pivot_offset_samples(offset_samples, selection_mode="middle_end")
            selection_mode = "middle_end"
            selected_labels = [s["label"] for s in selected_result["selected_samples"]]

        pivot_offset_x_px = float(selected_result["pivot_offset_x_px_active"])
        pivot_offset_y_px = float(selected_result["pivot_offset_y_px_active"])

        print("\nActive pivot offset selection:")
        print(f"  selection_mode = {selection_mode}")
        print(f"  selected_labels = {selected_labels}")
        print(f"  active dx [px] = {pivot_offset_x_px:+.3f}")
        print(f"  active dy [px] = {pivot_offset_y_px:+.3f}")

        pivot_offset_info = {
            "mode": "clicked_true_pivot_samples",
            "samples": offset_samples,
            "selection_mode": selection_mode,
            "selected_labels": selected_labels,
            "selected_samples": selected_result["selected_samples"],
            "pivot_offset_x_px_active": pivot_offset_x_px,
            "pivot_offset_y_px_active": pivot_offset_y_px,

            # legacy-compatible fields
            "pivot_offset_x_px": pivot_offset_x_px,
            "pivot_offset_y_px": pivot_offset_y_px
        }
        
    elif pivot_choice == "4":
        n_samples_txt = input("How many pivot samples across travel? [default 7]: ").strip()
        n_samples = int(n_samples_txt) if n_samples_txt else 7
        n_samples = max(3, min(16, n_samples))

        early_samples_txt = input(
            "How many of those samples should be in the first 5 seconds? [default 4]: "
        ).strip()
        early_samples = int(early_samples_txt) if early_samples_txt else min(4, n_samples - 1)
        early_samples = max(1, min(n_samples - 1, early_samples))

        manual_txt = input(
            "Optional: enter custom sample indices comma-separated, or press Enter to auto-generate: "
        ).strip()

        if manual_txt:
            sample_indices = sorted(
                set(int(v.strip()) for v in manual_txt.split(",") if v.strip())
            )
        else:
            sample_indices = get_biased_pivot_sample_indices(
                frame_count=frame_count,
                fps_used=fps_used,
                total_samples=n_samples,
                early_time_s=5.0,
                early_samples=early_samples,
                start_margin_frames=200,
                end_margin_frames=200
            )

        print("\nCaching representative frames for x-dependent pivot model...")
        print(f"  Sample frame indices: {sample_indices}")

        sample_frames = read_selected_roi_frames_sequential(video_path, roi_xywh, sample_indices)
        offset_samples = []

        print("\nCollecting pivot samples for x-dependent model:")

        for k, frame_idx_rep in enumerate(sample_indices, start=1):
            rep_frame = sample_frames.get(frame_idx_rep, None)
            if rep_frame is None:
                print(f"  WARNING: Could not read frame {frame_idx_rep}. Skipping.")
                continue

            green_ctr_auto, _, _, _ = detect_marker_on_roi_frame(rep_frame, green_hsv, tracking_defaults)

            if green_ctr_auto is None:
                print(f"  WARNING: Could not auto-detect green marker on frame {frame_idx_rep}.")
                print("  You can manually click the green marker center.")
            
            green_ctr_rep = click_marker_center_confirmable(
                window_name=f"Confirm GREEN marker sample {k}",
                image_bgr=rep_frame,
                detected_ctr_xy=green_ctr_auto,
                prompt_text=f"sample {k}/{len(sample_indices)} frame {frame_idx_rep}: confirm GREEN marker center",
                marker_color_bgr=(0, 255, 0),
                max_display_width=900,
                max_display_height=600
            )
            
            if green_ctr_rep is None:
                print("Cancelled.")
                return
            
            green_ctr_rep = (float(green_ctr_rep[0]), float(green_ctr_rep[1]))
            
            temp = rep_frame.copy()
            green_ctr_i = (int(round(green_ctr_rep[0])), int(round(green_ctr_rep[1])))
            cv2.circle(temp, green_ctr_i, 8, (0, 255, 0), 2)
            cv2.circle(temp, green_ctr_i, 3, (0, 255, 0), -1)
            cv2.putText(
                temp,
                f"sample {k}/{len(sample_indices)} frame {frame_idx_rep}: click TRUE pivot",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            true_pivot_pt = click_true_pivot_confirmable(
            window_name=f"TRUE pivot sample {k}",
            image_bgr=rep_frame,
            green_ctr_xy=green_ctr_rep,
            prompt_text=f"sample {k}/{len(sample_indices)} frame {frame_idx_rep}: click TRUE pivot",
            max_display_width=900,
            max_display_height=600,
            warning_offset_px=80.0
        )
        
            if true_pivot_pt is None:
                print("Cancelled.")
                return
            
            dx = float(true_pivot_pt[0] - green_ctr_rep[0])
            dy = float(true_pivot_pt[1] - green_ctr_rep[1])
            
            print(
                f"  Accepted sample {k}: "
                f"x: {format_signed_sum(green_ctr_rep[0], dx)} px, "
                f"y: {format_signed_sum(green_ctr_rep[1], dy)} px"
            )
                
            
            offset_samples.append({
                "label": f"sample_{k}",
                "frame_idx": int(frame_idx_rep),
                "green_marker_center_roi_px": [float(green_ctr_rep[0]), float(green_ctr_rep[1])],
                "true_pivot_roi_px": [int(true_pivot_pt[0]), int(true_pivot_pt[1])],
                "pivot_offset_x_px": dx,
                "pivot_offset_y_px": dy
            })

        if len(offset_samples) < 2:
            print("ERROR: Need at least 2 valid pivot samples for x-dependent model.")
            return
        
        print_pivot_sample_summary(offset_samples)
        
        pivot_model = fit_piecewise_linear_pivot_model(offset_samples)

        print("\nX-dependent pivot model nodes:")
        for xg, dx, dy in zip(
            pivot_model["x_green_nodes_px"],
            pivot_model["dx_nodes_px"],
            pivot_model["dy_nodes_px"]
        ):
            x_true = xg + dx
            print(
                f"  x_green={xg:.3f} px -> "
                f"dx={dx:+.3f} px, dy={dy:+.3f} px | "
                f"x_true: {format_signed_sum(xg, dx)} px"
            )
        # backward-compatible constant fallback = mean of nodes
        pivot_offset_x_px = float(np.mean(pivot_model["dx_nodes_px"]))
        pivot_offset_y_px = float(np.mean(pivot_model["dy_nodes_px"]))

        pivot_offset_info = {
            "mode": "clicked_true_pivot_x_model",
            "samples": offset_samples,
            "model": pivot_model,

            # backward-compatible fallback fields
            "pivot_offset_x_px_active": pivot_offset_x_px,
            "pivot_offset_y_px_active": pivot_offset_y_px,
            "pivot_offset_x_px": pivot_offset_x_px,
            "pivot_offset_y_px": pivot_offset_y_px
        }

    else:
            pivot_offset_info = {
            "mode": "marker_center_is_pivot",
            "green_marker_center_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "true_pivot_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "pivot_offset_x_px_active": 0.0,
            "pivot_offset_y_px_active": 0.0,
            "pivot_offset_x_px": 0.0,
            "pivot_offset_y_px": 0.0
    }
        

    # --------------------------------------------------------
    # Build masks on first frame for preview only
    # --------------------------------------------------------
    green_mask_raw = apply_hsv_ranges(roi_hsv, green_hsv)
    pink_mask_raw = apply_hsv_ranges(roi_hsv, pink_hsv)

    green_mask = clean_mask(
        green_mask_raw,
        open_kernel=int(tracking_defaults.get("morph_open_kernel", 2)),
        close_kernel=int(tracking_defaults.get("morph_close_kernel", 5))
    )
    pink_mask = clean_mask(
        pink_mask_raw,
        open_kernel=int(tracking_defaults.get("morph_open_kernel", 2)),
        close_kernel=int(tracking_defaults.get("morph_close_kernel", 5))
    )

    green_preview = preview_mask_on_frame(roi_frame, green_mask)
    pink_preview = preview_mask_on_frame(roi_frame, pink_mask)

    gx0, gy0, gx1, gy1 = green_box
    px0, py0, px1, py1 = pink_box

    cv2.rectangle(green_preview, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
    cv2.circle(green_preview, green_pt, 5, (0, 255, 0), -1)

    if pivot_choice in ("2", "3", "4"):
        green_ctr_first, _, _, _ = detect_marker_on_roi_frame(roi_frame, green_hsv, tracking_defaults)
        if green_ctr_first is not None:
            pivot_vis = (
                int(round(green_ctr_first[0] + pivot_offset_x_px)),
                int(round(green_ctr_first[1] + pivot_offset_y_px))
            )
            cv2.circle(green_preview, pivot_vis, 5, (0, 255, 255), -1)
            cv2.line(
                green_preview,
                (int(round(green_ctr_first[0])), int(round(green_ctr_first[1]))),
                pivot_vis,
                (255, 255, 255),
                2
            )

    cv2.rectangle(pink_preview, (px0, py0), (px1, py1), (255, 0, 255), 2)
    cv2.circle(pink_preview, pink_pt, 5, (255, 0, 255), -1)
    green_preview_disp, _ = resize_for_display(green_preview, 900, 600)
    pink_preview_disp, _ = resize_for_display(pink_preview, 900, 600)
    
    cv2.imshow("Preview GREEN cleaned mask", green_preview_disp)
    cv2.imshow("Preview PINK cleaned mask", pink_preview_disp)
    # cv2.imshow("Preview GREEN cleaned mask", green_preview)
    # cv2.imshow("Preview PINK cleaned mask", pink_preview)
    print("\nInspect the preview windows. Press any key in an image window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nMarker HSV summary:")
    print(f"  GREEN median HSV: {green_hsv['median_hsv']}")
    print(f"  PINK  median HSV: {pink_hsv['median_hsv']}")

    # --------------------------------------------------------
    # Scale options
    # --------------------------------------------------------
    print("\nScale calibration options:")
    print("  1) Use known pendulum rod length")
    print("  2) Use another reference length later")
    print("  3) Use 1-inch marker diameter(s) for px/in with multi-window averaging")
    print("  4) Use markers + 1-inch red tape length for combined px/in")
    scale_choice = input("Choose 1, 2, 3, or 4 [default 1]: ").strip()

    length_info = {}

    if scale_choice == "2":
        ref_name = input("Enter name of reference distance (e.g. ruler segment): ").strip()
        ref_len = float(input("Enter physical length in inches: ").strip())

        length_info["mode"] = "reference_length"
        length_info["reference_name"] = ref_name
        length_info["reference_length_in"] = ref_len

    elif scale_choice in ("3", "4"):
        green_window_stats = {}
        pink_window_stats = {}

        green_all_eq = []
        pink_all_eq = []

        print("\nOption A: multi-window marker-diameter px/in")
        print("Using windows:")
        for label, idxs in frame_windows.items():
            print(f"  {label:<6}: frames {idxs[0]} to {idxs[-1]}")

        for label, idxs in frame_windows.items():
            gstats = collect_marker_diameter_stats(video_path, roi_xywh, idxs, green_hsv, tracking_defaults)
            pstats = collect_marker_diameter_stats(video_path, roi_xywh, idxs, pink_hsv, tracking_defaults)

            green_window_stats[label] = {
                "frame_indices": [int(i) for i in idxs],
                "summary_equivalent_diameter_px": gstats["summary_equivalent_diameter_px"],
                "summary_enclosing_circle_diameter_px": gstats["summary_enclosing_circle_diameter_px"],
                "summary_area_px2": gstats["summary_area_px2"]
            }
            pink_window_stats[label] = {
                "frame_indices": [int(i) for i in idxs],
                "summary_equivalent_diameter_px": pstats["summary_equivalent_diameter_px"],
                "summary_enclosing_circle_diameter_px": pstats["summary_enclosing_circle_diameter_px"],
                "summary_area_px2": pstats["summary_area_px2"]
            }

            green_all_eq.extend(gstats["equivalent_diameters_px"])
            pink_all_eq.extend(pstats["equivalent_diameters_px"])

        print("\nGREEN marker window summaries:")
        for label in ["start", "middle", "end"]:
            print_window_summary(label, {
                "summary_equivalent_diameter_px": green_window_stats[label]["summary_equivalent_diameter_px"]
            })

        print("\nPINK marker window summaries:")
        for label in ["start", "middle", "end"]:
            print_window_summary(label, {
                "summary_equivalent_diameter_px": pink_window_stats[label]["summary_equivalent_diameter_px"]
            })

        green_overall = summarize_values(green_all_eq)
        pink_overall = summarize_values(pink_all_eq)

        if green_overall["n_valid"] == 0 and pink_overall["n_valid"] == 0:
            print("ERROR: Could not estimate px/in from either marker across sampled windows.")
            return

        px_per_in_green = green_overall["median"] if green_overall["n_valid"] > 0 else None
        px_per_in_pink = pink_overall["median"] if pink_overall["n_valid"] > 0 else None
        
        marker_only_estimates = [
            v for v in [px_per_in_green, px_per_in_pink]
            if v is not None and np.isfinite(v)
        ]
        avg_px_per_in_markers = float(np.mean(marker_only_estimates))
        
        px_per_in_red_tape_length = None
        red_tape_hsv = None
        red_tape_window_stats = {}
        
        if scale_choice == "4":
            print("\nRed tape calibration:")
            print("Click the CENTER of the 1-inch-long red tape on the first frame.")
        
            red_click = click_point_on_image(
                "Click RED tape",
                roi_frame,
                "Click RED tape center",
                prompt_color=(0, 0, 255)
            )
        
            if red_click is not None:
                red_patch, _ = sample_patch(roi_hsv, red_click, half_size=6)
                red_tape_hsv = build_hsv_range_from_patch(red_patch)
        
                red_all_lengths = []
        
                for label, idxs in frame_windows.items():
                    rstats = collect_tape_length_stats(video_path, roi_xywh, idxs, red_tape_hsv, tracking_defaults)
        
                    red_tape_window_stats[label] = {
                        "frame_indices": [int(i) for i in idxs],
                        "summary_length_long_px": rstats["summary_length_long_px"],
                        "summary_width_short_px": rstats["summary_width_short_px"],
                        "summary_area_px2": rstats["summary_area_px2"]
                    }
        
                    red_all_lengths.extend(rstats["lengths_long_px"])
        
                red_overall = summarize_values(red_all_lengths)
        
                if red_overall["n_valid"] > 0:
                    px_per_in_red_tape_length = red_overall["median"]
                    print(f"  RED tape median long length : {px_per_in_red_tape_length:.3f} px/in")
                else:
                    print("  RED tape could not be measured reliably; skipping.")
            else:
                print("  Red tape click cancelled; skipping.")
        
        combined_estimates = list(marker_only_estimates)
        if px_per_in_red_tape_length is not None and np.isfinite(px_per_in_red_tape_length):
            combined_estimates.append(px_per_in_red_tape_length)
        
        avg_px_per_in = float(np.mean(combined_estimates))

        print("\nOverall scale estimates:")
        if px_per_in_green is not None:
            print(f"  GREEN marker median eq. diameter : {px_per_in_green:.3f} px/in")
        if px_per_in_pink is not None:
            print(f"  PINK  marker median eq. diameter : {px_per_in_pink:.3f} px/in")
        if px_per_in_red_tape_length is not None:
            print(f"  RED tape median long length      : {px_per_in_red_tape_length:.3f} px/in")

        print(f"  Marker-only average px/in        : {avg_px_per_in_markers:.3f}")
        print(f"  Combined average px/in           : {avg_px_per_in:.3f}")
        

        if avg_px_per_in > 0:
            pivot_offset_x_in = pivot_offset_x_px / avg_px_per_in
            pivot_offset_y_in = pivot_offset_y_px / avg_px_per_in
            print(f"  pivot_offset_x_in                 : {pivot_offset_x_in:.3f} in")
            print(f"  pivot_offset_y_in                 : {pivot_offset_y_in:.3f} in")
        else:
            pivot_offset_x_in = None
            pivot_offset_y_in = None

        # Save Option A
        length_info["mode"] = "marker_diameter_multiframe"
        length_info["marker_real_diameter_in"] = 1.0
        length_info["frame_windows"] = {k: [int(i) for i in v] for k, v in frame_windows.items()}
        
        length_info["green_window_stats"] = green_window_stats
        length_info["pink_window_stats"] = pink_window_stats
        length_info["green_overall_summary_equivalent_diameter_px"] = green_overall
        length_info["pink_overall_summary_equivalent_diameter_px"] = pink_overall
        
        length_info["px_per_in_green"] = px_per_in_green
        length_info["px_per_in_pink"] = px_per_in_pink
        length_info["px_per_in_avg_markers"] = avg_px_per_in_markers
        
        length_info["red_tape_hsv"] = red_tape_hsv
        length_info["red_tape_window_stats"] = red_tape_window_stats
        length_info["px_per_in_red_tape_length"] = px_per_in_red_tape_length
        
        # backward-compatible combined average used by existing analysis fallback
        length_info["px_per_in_avg"] = avg_px_per_in
        
        length_info["pivot_offset_x_in"] = pivot_offset_x_px / avg_px_per_in if avg_px_per_in > 0 else None
        length_info["pivot_offset_y_in"] = pivot_offset_y_px / avg_px_per_in if avg_px_per_in > 0 else None

        # ----------------------------------------------------
        # Option B sanity check: printed only, not saved
        # ----------------------------------------------------
        run_sanity_check = input(
            "\nRun Option B travel-based px/in sanity check using first/last 10-frame windows? [y/N]: "
        ).strip().lower()

        if run_sanity_check == "y":
            measured_travel_in_txt = input(
                "Enter physically measured trolley travel in inches (e.g. 23.5): "
            ).strip()

            if measured_travel_in_txt:
                measured_travel_in = float(measured_travel_in_txt)

                start_x_stats = collect_marker_x_stats(
                    video_path, roi_xywh, frame_windows["start"], green_hsv, tracking_defaults
                )
                end_x_stats = collect_marker_x_stats(
                    video_path, roi_xywh, frame_windows["end"], green_hsv, tracking_defaults
                )

                start_mean_x = start_x_stats["summary_x_px"]["mean"]
                end_mean_x = end_x_stats["summary_x_px"]["mean"]

                print("\nOption B sanity check (printed only, not saved):")
                if start_mean_x is None or end_mean_x is None:
                    print("  Could not compute start/end mean green x-position reliably.")
                elif measured_travel_in <= 0:
                    print("  Measured travel must be > 0.")
                else:
                    delta_x_px = abs(end_mean_x - start_mean_x)
                    px_per_in_travelcheck = delta_x_px / measured_travel_in

                    print(f"  Start 10-frame mean green x [px] : {start_mean_x:.3f}")
                    print(f"  End   10-frame mean green x [px] : {end_mean_x:.3f}")
                    print(f"  Net travel from windows    [px]  : {delta_x_px:.3f}")
                    print(f"  Measured physical travel   [in]  : {measured_travel_in:.3f}")
                    print(f"  Travel-based px/in check         : {px_per_in_travelcheck:.3f}")

                    if avg_px_per_in > 0:
                        pct_diff = 100.0 * (px_per_in_travelcheck - avg_px_per_in) / avg_px_per_in
                        print(f"  Percent diff vs Option A         : {pct_diff:+.2f}%")

    else:
        rod_len = float(input("Enter physical pendulum length in inches: ").strip())
        length_info["mode"] = "rod_length"
        length_info["rod_length_in"] = rod_len

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------
    analysis_defaults = {
    "theta_zero_method": "late_window",
    "theta_bias_t0_s": 20.0,
    "theta_bias_t1_s": 85.0,
    "theta_zero_n_frames": 20,
    "theta_manual_offset_deg": 0.0,

    "x_zero_n_frames": 20,
    "direction_window_n_frames": 20,
    "position_smoothing_window": 2,
    "speed_smoothing_window": 54,

    "pivot_scale_source": "green",
    "bob_scale_source": "pink",
    "fallback_scale_source": "avg",

    "theta_primary_candidate": "quad_basic",
    "command_end_time_s": 5.0,
    "theta_band_deg": 0.71,

    # Validation defaults added for updated analysis workflow
    "settling_deadband_dwell_s": 1.0,
    "command_speed_in_s": 4.6,
    "placement_goal_in": 0.125,
    "placement_poststop_buffer_s": 0.0,
    "no_motion_travel_threshold_in": 0.25,
    "stationary_noise_profile_path": "stationary_noise_profile.json",

    "velocity_smoothing_method": "savgol",
    "savgol_window": 7,
    "savgol_polyorder": 7,
    "speed_post_smoothing_window": 54,
    "use_stop_deadband": True,
    "stop_deadband_buffer_s": 0.3,
    "stop_deadband_sigma_mult": 2.2,
    "stop_deadband_final_pos_tol_in": 0.03,
    
    "metrics_min_peak_deg": 0.5,
    "metrics_min_peak_prom_deg": 0.15,
    "metrics_min_peak_spacing_s": 0.15,                                                 
    
    "use_poststop_theta_centering": True,
    "theta_poststop_center_delay_s": 2.0,
    "theta_poststop_center_min_samples": 100,
    "theta_poststop_center_max_abs_deg": 5.0,
    
    }

    config = {
        "video_path": video_path,
        "fps_metadata": fps_meta,
        "fps_used": fps_used,
        "roi_xywh": roi_xywh,
        "green_marker": {
            "name": "pivot",
            "initial_point_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "hsv": green_hsv
        },
        "pink_marker": {
            "name": "bob",
            "initial_point_roi_px": [int(pink_pt[0]), int(pink_pt[1])],
            "hsv": pink_hsv
        },
        "pivot_offset_info": pivot_offset_info,
        "length_info": length_info,
        "tracking_defaults": tracking_defaults,
        "analysis_defaults": analysis_defaults
    }

    out_name = input('\nEnter output config filename [default: config_dual_marker.json]: ').strip()
    if not out_name:
        out_name = "config_dual_marker.json"

    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved configuration to:\n  {os.path.abspath(out_name)}")
    print("\nDone.")


if __name__ == "__main__":
    main()