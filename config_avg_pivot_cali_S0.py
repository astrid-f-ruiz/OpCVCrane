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


def build_hsv_range_from_patch(hsv_patch, hue_pad=10, sat_pad=50, val_pad=50):
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


def clean_mask(mask, open_kernel=3, close_kernel=5):
    mask_out = mask.copy()

    if open_kernel and open_kernel > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, k_open)

    if close_kernel and close_kernel > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, k_close)

    return mask_out


def preview_mask_on_frame(frame_bgr, mask):
    overlay = frame_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def largest_contour(mask, min_area=20):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def contour_centroid(contour):
    if contour is None:
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


def click_point_on_image(window_name, image_bgr, prompt_text, prompt_color=(0, 255, 255)):
    global clicked_point
    clicked_point = None

    base = image_bgr.copy()
    cv2.putText(base, prompt_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, prompt_color, 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        disp = base.copy()
        if clicked_point is not None:
            cv2.circle(disp, clicked_point, 6, prompt_color, 2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if clicked_point is not None:
            pt = clicked_point
            cv2.destroyWindow(window_name)
            return pt

        if key == 27:
            cv2.destroyAllWindows()
            return None


def print_window_summary(title, stats):
    s = stats["summary_equivalent_diameter_px"]
    if s["n_valid"] == 0:
        print(f"  {title:<8}: no valid detections")
    else:
        print(
            f"  {title:<8}: n={s['n_valid']}, "
            f"mean={s['mean']:.3f}, median={s['median']:.3f}, std={s['std']:.3f} px/in"
        )


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
    roi = cv2.selectROI("Select ROI", frame_display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

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
    green_hsv = build_hsv_range_from_patch(green_patch)

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
        "min_contour_area_px": 30,
        "morph_open_kernel": 3,
        "morph_close_kernel": 5,
        "search_radius_px": 50,
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
    print("  3) Click true physical pivot point on representative start/middle/end frames and average")
    pivot_choice = input("Choose 1, 2, or 3 [default 1]: ").strip()

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

        pivot_offset_x_px = float(np.mean([s["pivot_offset_x_px"] for s in offset_samples]))
        pivot_offset_y_px = float(np.mean([s["pivot_offset_y_px"] for s in offset_samples]))

        print("\nPivot offset multi-frame average:")
        for s in offset_samples:
            print(
                f"  {s['label']:<6} frame {s['frame_idx']:>6}: "
                f"dx={s['pivot_offset_x_px']:+.3f} px, dy={s['pivot_offset_y_px']:+.3f} px"
            )
        print(f"  AVG dx [px] = {pivot_offset_x_px:+.3f}")
        print(f"  AVG dy [px] = {pivot_offset_y_px:+.3f}")

        pivot_offset_info = {
            "mode": "clicked_true_pivot_avg_start_middle_end",
            "samples": offset_samples,
            "pivot_offset_x_px": pivot_offset_x_px,
            "pivot_offset_y_px": pivot_offset_y_px
        }

    else:
        pivot_offset_info = {
            "mode": "marker_center_is_pivot",
            "green_marker_center_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "true_pivot_roi_px": [int(green_pt[0]), int(green_pt[1])],
            "pivot_offset_x_px": 0.0,
            "pivot_offset_y_px": 0.0
        }

    # --------------------------------------------------------
    # Build masks on first frame for preview only
    # --------------------------------------------------------
    green_mask_raw = apply_hsv_ranges(roi_hsv, green_hsv)
    pink_mask_raw = apply_hsv_ranges(roi_hsv, pink_hsv)

    green_mask = clean_mask(green_mask_raw, open_kernel=3, close_kernel=5)
    pink_mask = clean_mask(pink_mask_raw, open_kernel=3, close_kernel=5)

    green_preview = preview_mask_on_frame(roi_frame, green_mask)
    pink_preview = preview_mask_on_frame(roi_frame, pink_mask)

    gx0, gy0, gx1, gy1 = green_box
    px0, py0, px1, py1 = pink_box

    cv2.rectangle(green_preview, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
    cv2.circle(green_preview, green_pt, 5, (0, 255, 0), -1)

    if pivot_choice in ("2", "3"):
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

    cv2.imshow("Preview GREEN cleaned mask", green_preview)
    cv2.imshow("Preview PINK cleaned mask", pink_preview)
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
    scale_choice = input("Choose 1, 2, or 3 [default 1]: ").strip()

    length_info = {}

    if scale_choice == "2":
        ref_name = input("Enter name of reference distance (e.g. ruler segment): ").strip()
        ref_len = float(input("Enter physical length in inches: ").strip())

        length_info["mode"] = "reference_length"
        length_info["reference_name"] = ref_name
        length_info["reference_length_in"] = ref_len

    elif scale_choice == "3":
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

        px_estimates = [v for v in [px_per_in_green, px_per_in_pink] if v is not None and np.isfinite(v)]
        avg_px_per_in = float(np.mean(px_estimates))

        print("\nOverall marker-based scale estimates assuming each marker is 1.000 inch diameter:")
        if px_per_in_green is not None:
            print(f"  GREEN overall median eq. diameter : {px_per_in_green:.3f} px/in")
        else:
            print("  GREEN overall median eq. diameter : not available")

        if px_per_in_pink is not None:
            print(f"  PINK  overall median eq. diameter : {px_per_in_pink:.3f} px/in")
        else:
            print("  PINK  overall median eq. diameter : not available")

        print(f"  Average px/in used                : {avg_px_per_in:.3f}")

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
        length_info["px_per_in_avg"] = avg_px_per_in
        length_info["pivot_offset_x_in"] = pivot_offset_x_in
        length_info["pivot_offset_y_in"] = pivot_offset_y_in

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
        "theta_zero_method": "first_n_frames",
        "theta_zero_n_frames": 20,
        "theta_manual_offset_deg": 0.0,
        "speed_smoothing_window": 9
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