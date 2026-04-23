import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_with_header(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def moving_average_nan(x, window):
    x = np.asarray(x, dtype=float)

    if window is None or window <= 1:
        return x.copy()

    if window % 2 == 0:
        window += 1

    half = window // 2
    y = np.full_like(x, np.nan, dtype=float)

    for i in range(len(x)):
        i0 = max(0, i - half)
        i1 = min(len(x), i + half + 1)
        chunk = x[i0:i1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size > 0:
            y[i] = np.mean(valid)

    return y


def interpolate_nans(time_s, x):
    time_s = np.asarray(time_s, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(x)
    if np.sum(valid) < 2:
        return x.copy()

    xp = time_s[valid]
    fp = x[valid]
    return np.interp(time_s, xp, fp)


def gradient_with_time(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return np.full_like(y, np.nan, dtype=float)

    return np.gradient(y, t)


def mean_first_n_valid(x, valid_mask, n):
    x = np.asarray(x, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    vals = x[valid_mask]
    if vals.size == 0:
        return np.nan

    n_use = min(n, vals.size)
    return float(np.mean(vals[:n_use]))


def mean_last_n_valid(x, valid_mask, n):
    x = np.asarray(x, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    vals = x[valid_mask]
    if vals.size == 0:
        return np.nan

    n_use = min(n, vals.size)
    return float(np.mean(vals[-n_use:]))


def mean_in_time_window(x, t, valid_mask, t0, t1):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    mask = valid_mask & np.isfinite(x) & (t >= t0) & (t <= t1)
    if np.sum(mask) == 0:
        return np.nan

    return float(np.mean(x[mask]))


# ============================================================
# Scale handling
# ============================================================

def compute_px_per_in(cfg, raw):
    length_info = cfg.get("length_info", {})
    mode = length_info.get("mode", "")

    if mode in ("marker_diameter", "marker_diameter_multiframe"):
        px_per_in = length_info.get("px_per_in_avg", None)
        if px_per_in is None or px_per_in <= 0:
            raise ValueError(f"{mode} mode exists but px_per_in_avg is invalid.")
        return float(px_per_in), mode

    elif mode == "rod_length":
        rod_length_in = float(length_info["rod_length_in"])

        pivot_offset_x_px = float(cfg["pivot_offset_info"]["pivot_offset_x_px"])
        pivot_offset_y_px = float(cfg["pivot_offset_info"]["pivot_offset_y_px"])

        pivot_x = raw["pivot_x_roi"].astype(float)
        pivot_y = raw["pivot_y_roi"].astype(float)
        bob_x = raw["bob_x_roi"].astype(float)
        bob_y = raw["bob_y_roi"].astype(float)
        pivot_found = raw["pivot_found"].astype(int)
        bob_found = raw["bob_found"].astype(int)

        pivot_x_corr = pivot_x + pivot_offset_x_px
        pivot_y_corr = pivot_y + pivot_offset_y_px

        valid = (
            (pivot_found == 1) &
            (bob_found == 1) &
            np.isfinite(pivot_x_corr) &
            np.isfinite(pivot_y_corr) &
            np.isfinite(bob_x) &
            np.isfinite(bob_y)
        )

        dist_px = np.sqrt((bob_x - pivot_x_corr) ** 2 + (bob_y - pivot_y_corr) ** 2)

        theta_zero_n_frames = int(cfg.get("analysis_defaults", {}).get("theta_zero_n_frames", 20))
        dist0_px = mean_first_n_valid(dist_px, valid, theta_zero_n_frames)

        if not np.isfinite(dist0_px) or rod_length_in <= 0:
            raise ValueError("Could not estimate px/in from rod length.")

        px_per_in = dist0_px / rod_length_in
        return float(px_per_in), "rod_length_estimated_from_initial_distance"

    elif mode == "reference_length":
        raise ValueError(
            "reference_length mode is stored in config, but no pixel measurement for that reference "
            "has been implemented yet."
        )

    else:
        raise ValueError(f"Unsupported length_info mode: {mode}")


# ============================================================
# Main analysis
# ============================================================

def main():
    print("\n=== Dual Marker Analysis ===\n")

    config_path = input('Enter config JSON path [default: config_dual_marker.json]: ').strip().strip('"')
    if not config_path:
        config_path = "config_dual_marker.json"

    raw_csv_path = input('Enter raw tracking CSV path [default: raw_tracking.csv]: ').strip().strip('"')
    if not raw_csv_path:
        raw_csv_path = "raw_tracking.csv"

    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found:\n{config_path}")
        return

    if not os.path.isfile(raw_csv_path):
        print(f"ERROR: Raw CSV not found:\n{raw_csv_path}")
        return

    cfg = load_json(config_path)
    raw = load_csv_with_header(raw_csv_path)

    out_csv = input('Output analyzed CSV [default: analyzed_tracking.csv]: ').strip()
    if not out_csv:
        out_csv = "analyzed_tracking.csv"

    angle_plot_path = input('Angle plot filename [default: angle_plot.png]: ').strip()
    if not angle_plot_path:
        angle_plot_path = "angle_plot.png"

    pos_plot_path = input('Pivot position plot filename [default: pivot_position_plot.png]: ').strip()
    if not pos_plot_path:
        pos_plot_path = "pivot_position_plot.png"

    speed_plot_path = input('Pivot speed plot filename [default: pivot_speed_plot.png]: ').strip()
    if not speed_plot_path:
        speed_plot_path = "pivot_speed_plot.png"

    show_plots_txt = input("Show plots in Spyder/UI? [y/N]: ").strip().lower()
    show_plots = (show_plots_txt == "y")

    # --------------------------------------------------------
    # Raw columns
    # --------------------------------------------------------
    frame_idx = raw["frame_idx"].astype(int)
    time_s = raw["time_s"].astype(float)

    pivot_x = raw["pivot_x_roi"].astype(float)
    pivot_y = raw["pivot_y_roi"].astype(float)
    bob_x = raw["bob_x_roi"].astype(float)
    bob_y = raw["bob_y_roi"].astype(float)

    pivot_found = raw["pivot_found"].astype(int)
    bob_found = raw["bob_found"].astype(int)

    pivot_valid = (pivot_found == 1) & np.isfinite(pivot_x) & np.isfinite(pivot_y)
    theta_valid = (
        (pivot_found == 1) &
        (bob_found == 1) &
        np.isfinite(pivot_x) &
        np.isfinite(pivot_y) &
        np.isfinite(bob_x) &
        np.isfinite(bob_y)
    )

    # --------------------------------------------------------
    # Pivot correction
    # --------------------------------------------------------
    pivot_offset_x_px = float(cfg["pivot_offset_info"]["pivot_offset_x_px"])
    pivot_offset_y_px = float(cfg["pivot_offset_info"]["pivot_offset_y_px"])

    pivot_x_corr = pivot_x + pivot_offset_x_px
    pivot_y_corr = pivot_y + pivot_offset_y_px

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------
    try:
        px_per_in, px_source = compute_px_per_in(cfg, raw)
    except Exception as e:
        print(f"ERROR determining px/in: {e}")
        return

    print(f"\nUsing px/in = {px_per_in:.6f} from mode: {px_source}")

    # --------------------------------------------------------
    # Convert to inches
    # --------------------------------------------------------
    pivot_x_in_raw = pivot_x_corr / px_per_in
    pivot_y_in_raw = pivot_y_corr / px_per_in
    bob_x_in_raw = bob_x / px_per_in
    bob_y_in_raw = bob_y / px_per_in

    pivot_x_in_raw[~pivot_valid] = np.nan
    pivot_y_in_raw[~pivot_valid] = np.nan
    bob_x_in_raw[bob_found != 1] = np.nan
    bob_y_in_raw[bob_found != 1] = np.nan

    # --------------------------------------------------------
    # X zeroing and direction normalization
    # --------------------------------------------------------
    analysis_defaults = cfg.get("analysis_defaults", {})

    x_zero_n_frames = int(analysis_defaults.get("x_zero_n_frames", 20))
    direction_window_n_frames = int(analysis_defaults.get("direction_window_n_frames", 20))
    position_smoothing_window = int(analysis_defaults.get("position_smoothing_window", 31))
    speed_smoothing_window = int(analysis_defaults.get("speed_smoothing_window", 9))

    x0_in = mean_first_n_valid(pivot_x_in_raw, pivot_valid, x_zero_n_frames)
    if not np.isfinite(x0_in):
        x0_in = 0.0

    pivot_x_in_rel = pivot_x_in_raw - x0_in

    x_start = mean_first_n_valid(pivot_x_in_raw, pivot_valid, direction_window_n_frames)
    x_end = mean_last_n_valid(pivot_x_in_raw, pivot_valid, direction_window_n_frames)

    if np.isfinite(x_start) and np.isfinite(x_end):
        direction_sign = np.sign(x_end - x_start)
        if direction_sign == 0:
            direction_sign = 1.0
    else:
        direction_sign = 1.0

    pivot_x_in_report = direction_sign * pivot_x_in_rel

    # Smooth position first
    pivot_x_in_report_interp = interpolate_nans(time_s, pivot_x_in_report)
    pivot_x_in_report_smooth = moving_average_nan(pivot_x_in_report_interp, position_smoothing_window)

    # Speed from smoothed position
    pivot_vx_in_s_raw = gradient_with_time(pivot_x_in_report_interp, time_s)
    pivot_vx_in_s_report = gradient_with_time(pivot_x_in_report_smooth, time_s)
    pivot_vx_in_s_report_smooth = moving_average_nan(pivot_vx_in_s_report, speed_smoothing_window)

    # --------------------------------------------------------
    # Theta calculation
    # Keep theta sign convention unchanged
    # --------------------------------------------------------
    dx = bob_x - pivot_x_corr
    dy = bob_y - pivot_y_corr

    theta_raw_rad = np.full_like(time_s, np.nan, dtype=float)
    theta_raw_rad[theta_valid] = np.arctan2(dx[theta_valid], dy[theta_valid])
    theta_raw_deg = np.degrees(theta_raw_rad)

    theta_zero_method = analysis_defaults.get("theta_zero_method", "first_n_frames")
    theta_zero_n_frames = int(analysis_defaults.get("theta_zero_n_frames", 20))
    theta_manual_offset_deg = float(analysis_defaults.get("theta_manual_offset_deg", 0.0))

    if theta_zero_method == "first_n_frames":
        theta_bias_deg = mean_first_n_valid(theta_raw_deg, theta_valid, theta_zero_n_frames)

    elif theta_zero_method == "late_window":
        late_t0 = float(analysis_defaults.get("theta_bias_t0_s", 10.0))
        late_t1 = float(analysis_defaults.get("theta_bias_t1_s", 60.0))
        theta_bias_deg = mean_in_time_window(theta_raw_deg, time_s, theta_valid, late_t0, late_t1)

        if not np.isfinite(theta_bias_deg):
            print("WARNING: late_window theta bias failed, falling back to first_n_frames.")
            theta_bias_deg = mean_first_n_valid(theta_raw_deg, theta_valid, theta_zero_n_frames)

    elif theta_zero_method == "none":
        theta_bias_deg = 0.0

    else:
        print(f"WARNING: Unknown theta_zero_method '{theta_zero_method}', using first_n_frames.")
        theta_bias_deg = mean_first_n_valid(theta_raw_deg, theta_valid, theta_zero_n_frames)

    if not np.isfinite(theta_bias_deg):
        theta_bias_deg = 0.0

    theta_zeroed_deg = theta_raw_deg - theta_bias_deg - theta_manual_offset_deg

    # --------------------------------------------------------
    # Summary values
    # --------------------------------------------------------
    travel_measured_in = mean_last_n_valid(pivot_x_in_report, pivot_valid, direction_window_n_frames)
    max_abs_theta_deg = np.nanmax(np.abs(theta_zeroed_deg)) if np.any(np.isfinite(theta_zeroed_deg)) else np.nan
    max_speed_in_s = np.nanmax(np.abs(pivot_vx_in_s_report_smooth)) if np.any(np.isfinite(pivot_vx_in_s_report_smooth)) else np.nan

    print("\nAnalysis settings:")
    print(f"  x_zero_n_frames           : {x_zero_n_frames}")
    print(f"  direction_window_n_frames : {direction_window_n_frames}")
    print(f"  direction_sign            : {direction_sign:+.0f}")
    print(f"  position_smoothing_window : {position_smoothing_window}")
    print(f"  speed_smoothing_window    : {speed_smoothing_window}")
    print(f"  theta_zero_method         : {theta_zero_method}")
    print(f"  theta bias used [deg]     : {theta_bias_deg:.6f}")
    print(f"  theta manual offset [deg] : {theta_manual_offset_deg:.6f}")

    if np.isfinite(travel_measured_in):
        print(f"  reported net travel [in]  : {travel_measured_in:.6f}")

    # --------------------------------------------------------
    # Save analyzed CSV
    # --------------------------------------------------------
    header = ",".join([
        "frame_idx",
        "time_s",
        "pivot_x_roi",
        "pivot_y_roi",
        "bob_x_roi",
        "bob_y_roi",
        "pivot_found",
        "bob_found",
        "pivot_x_corr_roi",
        "pivot_y_corr_roi",
        "pivot_x_in_raw",
        "pivot_y_in_raw",
        "pivot_x_in_rel",
        "pivot_x_in_report",
        "pivot_x_in_report_smooth",
        "bob_x_in_raw",
        "bob_y_in_raw",
        "theta_raw_deg",
        "theta_zeroed_deg",
        "pivot_vx_in_s_raw",
        "pivot_vx_in_s_report",
        "pivot_vx_in_s_report_smooth",
        "direction_sign"
    ])

    out_matrix = np.column_stack([
        frame_idx,
        time_s,
        pivot_x,
        pivot_y,
        bob_x,
        bob_y,
        pivot_found,
        bob_found,
        pivot_x_corr,
        pivot_y_corr,
        pivot_x_in_raw,
        pivot_y_in_raw,
        pivot_x_in_rel,
        pivot_x_in_report,
        pivot_x_in_report_smooth,
        bob_x_in_raw,
        bob_y_in_raw,
        theta_raw_deg,
        theta_zeroed_deg,
        pivot_vx_in_s_raw,
        pivot_vx_in_s_report,
        pivot_vx_in_s_report_smooth,
        np.full_like(time_s, direction_sign, dtype=float)
    ])

    np.savetxt(out_csv, out_matrix, delimiter=",", header=header, comments="", fmt="%.10f")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time_s, theta_zeroed_deg)
    plt.xlabel("Time [s]")
    plt.ylabel("Sway angle [deg]")
    plt.title("Pendulum Sway Angle vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(angle_plot_path, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, pivot_x_in_report)
    plt.plot(time_s, pivot_x_in_report_smooth, label="smoothed")
    plt.xlabel("Time [s]")
    plt.ylabel("Pivot x-position [in]")
    plt.title("Pivot Horizontal Position vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pos_plot_path, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, pivot_vx_in_s_raw, label="raw from interpolated position")
    plt.plot(time_s, pivot_vx_in_s_report, label="from smoothed position")
    plt.plot(time_s, pivot_vx_in_s_report_smooth, label="final smoothed")
    plt.xlabel("Time [s]")
    plt.ylabel("Pivot horizontal speed [in/s]")
    plt.title("Pivot Horizontal Speed vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(speed_plot_path, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    print("\nSaved analyzed CSV:")
    print(f"  {os.path.abspath(out_csv)}")

    print("\nSaved plots:")
    print(f"  {os.path.abspath(angle_plot_path)}")
    print(f"  {os.path.abspath(pos_plot_path)}")
    print(f"  {os.path.abspath(speed_plot_path)}")

    print("\nQuick summary:")
    print(f"  Max |theta| [deg]         : {max_abs_theta_deg:.6f}")
    print(f"  Max |pivot vx| [in/s]     : {max_speed_in_s:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()