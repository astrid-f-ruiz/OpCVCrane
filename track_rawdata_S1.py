import cv2
import csv
import json
import math
import numpy as np
import os
import re


# ============================================================
# Utility helpers
# ============================================================

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def derive_raw_output_path_from_config_path(config_path):
    """
    Match the current analysis-code default.

    Example:
      H_CL_12IN_T7.json -> H_CL_12IN_T_RAW7
      H_OL_21IN_T1.json -> H_OL_21IN_T_RAW1
      M_OL_38LB_T6.json -> M_OL_38LB_T_RAW6

    No .csv extension, because the analysis code currently defaults to extensionless files.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    stem = os.path.splitext(os.path.basename(config_path))[0]

    m = re.match(r"^(.*?)(\d+)$", stem)
    if m:
        prefix = m.group(1)
        trial_num = m.group(2)
        raw_name = f"{prefix}_RAW{trial_num}"
    else:
        raw_name = f"{stem}_RAW"

    return os.path.join(config_dir, raw_name)

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


def clean_mask(mask, open_kernel=3, close_kernel=5):
    mask_out = mask.copy()

    if open_kernel and open_kernel > 1:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, k_open)

    if close_kernel and close_kernel > 1:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, k_close)

    return mask_out


def contour_centroid(contour):
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-9:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (float(cx), float(cy))


def distance_xy(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clip_search_window(center_xy, radius_px, width, height):
    cx, cy = center_xy
    x0 = max(0, int(round(cx - radius_px)))
    x1 = min(width, int(round(cx + radius_px)))
    y0 = max(0, int(round(cy - radius_px)))
    y1 = min(height, int(round(cy + radius_px)))
    return x0, y0, x1, y1


def select_best_contour(mask, last_xy=None, min_area=20):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area:
            ctr = contour_centroid(c)
            if ctr is not None:
                valid.append((c, a, ctr))

    if not valid:
        return None, None, None

    if last_xy is None:
        # If no prior, pick largest area
        c, a, ctr = max(valid, key=lambda t: t[1])
        return c, a, ctr

    # Otherwise prefer nearest to last known point
    c, a, ctr = min(valid, key=lambda t: distance_xy(t[2], last_xy))
    return c, a, ctr


def detect_marker(
    hsv_roi,
    hsv_range_dict,
    last_xy,
    min_area,
    morph_open_kernel,
    morph_close_kernel,
    search_radius_px,
    max_jump_px
):
    """
    Detection strategy:
      1) threshold full ROI
      2) if last position exists, try local search window first
      3) if local fails, fallback to full ROI
      4) reject jump if too far from last known position
    """
    full_mask_raw = apply_hsv_ranges(hsv_roi, hsv_range_dict)
    full_mask = clean_mask(full_mask_raw, morph_open_kernel, morph_close_kernel)

    h, w = full_mask.shape[:2]

    chosen_contour = None
    chosen_area = None
    chosen_ctr = None
    used_local = False

    # First: local search around last known point
    if last_xy is not None:
        x0, y0, x1, y1 = clip_search_window(last_xy, search_radius_px, w, h)

        if (x1 > x0) and (y1 > y0):
            local_mask = full_mask[y0:y1, x0:x1]
            c, a, ctr_local = select_best_contour(local_mask, last_xy=None, min_area=min_area)

            if c is not None and ctr_local is not None:
                ctr_global = (ctr_local[0] + x0, ctr_local[1] + y0)
                jump = distance_xy(ctr_global, last_xy)

                if jump <= max_jump_px:
                    chosen_contour = c
                    chosen_area = a
                    chosen_ctr = ctr_global
                    used_local = True

    # Fallback: search full ROI
    if chosen_ctr is None:
        c, a, ctr = select_best_contour(full_mask, last_xy=last_xy, min_area=min_area)

        if c is not None and ctr is not None:
            if last_xy is None:
                chosen_contour = c
                chosen_area = a
                chosen_ctr = ctr
            else:
                jump = distance_xy(ctr, last_xy)
                if jump <= max_jump_px or last_xy is None:
                    chosen_contour = c
                    chosen_area = a
                    chosen_ctr = ctr

    found = chosen_ctr is not None

    debug = {
        "mask": full_mask,
        "found": found,
        "centroid": chosen_ctr,
        "area": chosen_area,
        "used_local": used_local
    }

    return chosen_ctr, debug


def draw_marker_overlay(frame_roi_bgr, name, ctr, found, color_bgr, area=None):
    if found and ctr is not None:
        cx, cy = int(round(ctr[0])), int(round(ctr[1]))
        cv2.circle(frame_roi_bgr, (cx, cy), 8, color_bgr, 2)
        cv2.circle(frame_roi_bgr, (cx, cy), 2, color_bgr, -1)
        label = name
        if area is not None:
            label += f" A={area:.0f}"
        cv2.putText(
            frame_roi_bgr, label, (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2
        )
    else:
        cv2.putText(
            frame_roi_bgr, f"{name}: LOST", (20, 40 if name == "pivot" else 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2
        )


# ============================================================
# Main tracking
# ============================================================

def main():
    print("\n=== Dual Marker Raw Tracker ===\n")

    config_path = input(
        'Enter config JSON path [default: config_dual_marker.json]: '
    ).strip().strip('"')
    if not config_path:
        config_path = "config_dual_marker.json"

    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found:\n{config_path}")
        return

    cfg = load_config(config_path)

    video_path = cfg["video_path"]
    if not os.path.isfile(video_path):
        print(f"ERROR: Video file from config not found:\n{video_path}")
        return

    fps_used = float(cfg["fps_used"])
    x_roi, y_roi, w_roi, h_roi = cfg["roi_xywh"]

    green_cfg = cfg["green_marker"]
    pink_cfg = cfg["pink_marker"]

    tracking_defaults = cfg.get("tracking_defaults", {})
    min_area = int(tracking_defaults.get("min_contour_area_px", 150))
    morph_open_kernel = int(tracking_defaults.get("morph_open_kernel", 3))
    morph_close_kernel = int(tracking_defaults.get("morph_close_kernel", 5))
    search_radius_px = int(tracking_defaults.get("search_radius_px", 50))
    max_jump_px = int(tracking_defaults.get("max_jump_px", 60))

    pivot_last = tuple(map(float, green_cfg["initial_point_roi_px"]))
    bob_last = tuple(map(float, pink_cfg["initial_point_roi_px"]))

    show_preview_txt = input("Show live preview? [Y/n]: ").strip().lower()
    show_preview = (show_preview_txt != "n")
    
    preview_every_n_frames = 240

    output_csv = derive_raw_output_path_from_config_path(config_path)
    
    print("\nRAW tracking output will be saved to:")
    print(f"  {output_csv}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nTracking video: {video_path}")
    print(f"Using FPS: {fps_used:.3f}")
    print(f"Frame count: {frame_count}")
    print(f"ROI: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi}")

    rows = []

    frame_idx = 0
    pivot_lost_count = 0
    bob_lost_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Crop ROI
        frame_roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
        hsv_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)

        # ============================================================
        # Lost-frame recovery
        # ============================================================
        # If the marker has been lost for several frames, the previous
        # position may be stale. Temporarily allow full-ROI reacquisition.
        reacquire_after_lost_frames = int(tracking_defaults.get("reacquire_after_lost_frames", 5))
        
        pivot_search_last = None if pivot_lost_count >= reacquire_after_lost_frames else pivot_last
        bob_search_last = None if bob_lost_count >= reacquire_after_lost_frames else bob_last
        
        # Detect markers
        pivot_ctr, pivot_dbg = detect_marker(
            hsv_roi=hsv_roi,
            hsv_range_dict=green_cfg["hsv"],
            last_xy=pivot_search_last,
            min_area=min_area,
            morph_open_kernel=morph_open_kernel,
            morph_close_kernel=morph_close_kernel,
            search_radius_px=search_radius_px,
            max_jump_px=max_jump_px
        )
        
        bob_ctr, bob_dbg = detect_marker(
            hsv_roi=hsv_roi,
            hsv_range_dict=pink_cfg["hsv"],
            last_xy=bob_search_last,
            min_area=min_area,
            morph_open_kernel=morph_open_kernel,
            morph_close_kernel=morph_close_kernel,
            search_radius_px=search_radius_px,
            max_jump_px=max_jump_px
        )

        pivot_found = 1 if pivot_ctr is not None else 0
        bob_found = 1 if bob_ctr is not None else 0

        if pivot_found:
            pivot_last = pivot_ctr
            pivot_lost_count = 0
            pivot_x, pivot_y = pivot_ctr
        else:
            pivot_lost_count += 1
            pivot_x, pivot_y = np.nan, np.nan

        if bob_found:
            bob_last = bob_ctr
            bob_lost_count = 0
            bob_x, bob_y = bob_ctr
        else:
            bob_lost_count += 1
            bob_x, bob_y = np.nan, np.nan

        time_s = frame_idx / fps_used

        rows.append([
            frame_idx,
            time_s,
            pivot_x,
            pivot_y,
            bob_x,
            bob_y,
            pivot_found,
            bob_found,
            pivot_dbg["area"] if pivot_dbg["area"] is not None else np.nan,
            bob_dbg["area"] if bob_dbg["area"] is not None else np.nan
        ])

        # Optional preview
        if show_preview and (frame_idx % preview_every_n_frames == 0):

            display = frame_roi.copy()
        
            draw_marker_overlay(
                display, "pivot", pivot_ctr, pivot_found == 1,
                (0, 255, 0), area=pivot_dbg["area"]
            )
            draw_marker_overlay(
                display, "bob", bob_ctr, bob_found == 1,
                (255, 0, 255), area=bob_dbg["area"]
            )

            if pivot_found and bob_found:
                p1 = (int(round(pivot_ctr[0])), int(round(pivot_ctr[1])))
                p2 = (int(round(bob_ctr[0])), int(round(bob_ctr[1])))
                cv2.line(display, p1, p2, (255, 255, 255), 2)
        
                dx = bob_ctr[0] - pivot_ctr[0]
                dy = bob_ctr[1] - pivot_ctr[1]
                theta_raw_deg = math.degrees(math.atan2(dx, dy))
                cv2.putText(
                    display, f"raw theta = {theta_raw_deg:+.2f} deg",
                    (20, h_roi - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
        
            cv2.putText(
                display, f"frame {frame_idx}/{frame_count}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                display, f"t = {time_s:.3f} s",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                display, f"pivot lost streak = {pivot_lost_count}",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                display, f"bob lost streak = {bob_lost_count}",
                (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2
            )
        
            preview_scale = 0.7
            display_small = cv2.resize(
                display, None, fx=preview_scale, fy=preview_scale,
                interpolation=cv2.INTER_AREA
            )
        
            cv2.imshow("Dual Marker Tracking", display_small)



            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\nStopped by user (ESC).")
                break

        frame_idx += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_s",
            "pivot_x_roi",
            "pivot_y_roi",
            "bob_x_roi",
            "bob_y_roi",
            "pivot_found",
            "bob_found",
            "pivot_area_px2",
            "bob_area_px2"
        ])
        writer.writerows(rows)

    print(f"\nSaved raw tracking CSV to:\n  {os.path.abspath(output_csv)}")
    print(f"Tracked frames written: {len(rows)}")
    print("\nDone.")


if __name__ == "__main__":
    main()