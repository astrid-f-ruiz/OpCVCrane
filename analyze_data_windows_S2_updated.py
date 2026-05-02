import csv
import json
import os
import re
import shutil

import matplotlib.pyplot as plt

plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

import numpy as np
from scipy.signal import savgol_filter, find_peaks

# ============================================================
# Basic helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_tagged_name_from_stem(path_or_name, tag):
    """
    Examples:
      M_20_10LB_TZ3   -> M_20_10LB_TZ_RAW3
      I_15_5S_TZ12    -> I_15_5S_TZ_AP12
      XX_30IN_TZ4     -> XX_30IN_TZ_OUT4

    If no trailing digits exist:
      SAMPLE_NAME -> SAMPLE_NAME_RAW
    """
    folder = os.path.dirname(path_or_name)
    stem = os.path.splitext(os.path.basename(path_or_name))[0]

    m = re.match(r"^(.*?)(\d+)$", stem)
    if m:
        prefix = m.group(1)
        trial_num = m.group(2)
        out_name = f"{prefix}{tag}{trial_num}"
    else:
        out_name = f"{stem}_{tag}"

    return os.path.join(folder, out_name) if folder else out_name

def make_tagged_name_from_trial_stem(path_or_name, tag):
    """
    Examples:
      H_OL_21IN_T1   -> H_OL_21IN_T_RAW1
      M_20_10LB_T12  -> M_20_10LB_T_OUT12
      I_15_5S_T3     -> I_15_5S_T_AP3

    Keeps the same folder as the config path.
    If no trailing digits exist, falls back to stem + tag.
    """
    folder = os.path.dirname(path_or_name)
    stem = os.path.splitext(os.path.basename(path_or_name))[0]

    m = re.match(r"^(.*?)(\d+)$", stem)
    if m:
        prefix = m.group(1)
        trial_num = m.group(2)
        out_name = f"{prefix}{tag}{trial_num}"
    else:
        out_name = f"{stem}{tag}"

    return os.path.join(folder, out_name) if folder else out_name

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

def estimate_poststop_theta_center_bias(
    time_s,
    theta_deg,
    theta_valid,
    move_end_s,
    delay_s=2.0,
    min_samples=100,
    max_abs_deg=5.0
):
    """
    Estimate a single constant theta center bias after motion ends.

    This is meant to remove a small DC offset from the theta signal,
    not to remove real oscillation. It uses a robust median from the
    post-stop portion of the trial.

    Returns:
      bias_deg, info_dict
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)
    valid = np.asarray(theta_valid, dtype=bool)

    if not np.isfinite(move_end_s):
        return 0.0, {
            "method": "skipped_invalid_move_end",
            "theta_poststop_center_bias_deg": 0.0,
            "n_samples": 0
        }

    t0 = float(move_end_s) + float(delay_s)

    mask = (
        valid
        & np.isfinite(t)
        & np.isfinite(th)
        & (t >= t0)
        & (np.abs(th) <= float(max_abs_deg))
    )

    vals = th[mask]

    if vals.size < int(min_samples):
        return 0.0, {
            "method": "skipped_too_few_samples",
            "theta_poststop_center_bias_deg": 0.0,
            "n_samples": int(vals.size),
            "t0_s": t0
        }

    # Robust median center
    med = float(np.nanmedian(vals))

    # Optional outlier rejection around the median using MAD
    abs_dev = np.abs(vals - med)
    mad = float(np.nanmedian(abs_dev))

    if np.isfinite(mad) and mad > 1e-9:
        robust_sigma = 1.4826 * mad
        keep = abs_dev <= max(3.0 * robust_sigma, 0.25)
        vals2 = vals[keep]
        if vals2.size >= int(min_samples):
            med = float(np.nanmedian(vals2))
            vals = vals2

    return med, {
        "method": "poststop_theta_median_center",
        "theta_poststop_center_bias_deg": med,
        "n_samples": int(vals.size),
        "t0_s": t0,
        "delay_s": float(delay_s),
        "max_abs_deg": float(max_abs_deg)
    }

def make_valid_savgol_window(n, desired_window, polyorder):
    w = int(desired_window)

    min_w = polyorder + 2
    if w < min_w:
        w = min_w

    if w % 2 == 0:
        w += 1

    if w > n:
        w = n if n % 2 == 1 else n - 1

    if w < min_w or w < 3:
        return None

    return w


# ============================================================
# Scale handling
# ============================================================

def get_pivot_offset_arrays_px(cfg, pivot_x_roi):
    """
    Returns per-frame dx, dy pivot offsets in pixels.

    If calibration saved an x-dependent pivot model, use piecewise-linear interpolation.
    Otherwise fall back to the active constant offset.
    """
    pivot_x_roi = np.asarray(pivot_x_roi, dtype=float)
    info = cfg.get("pivot_offset_info", {})
    mode = info.get("mode", "")

    if mode == "clicked_true_pivot_x_model" and "model" in info:
        model = info["model"]
        x_nodes = np.asarray(model["x_green_nodes_px"], dtype=float)
        dx_nodes = np.asarray(model["dx_nodes_px"], dtype=float)
        dy_nodes = np.asarray(model["dy_nodes_px"], dtype=float)

        dx_arr = np.interp(pivot_x_roi, x_nodes, dx_nodes, left=dx_nodes[0], right=dx_nodes[-1])
        dy_arr = np.interp(pivot_x_roi, x_nodes, dy_nodes, left=dy_nodes[0], right=dy_nodes[-1])
        return dx_arr, dy_arr, "x_model_piecewise_linear"

    dx_const = float(info.get("pivot_offset_x_px_active", info.get("pivot_offset_x_px", 0.0)))
    dy_const = float(info.get("pivot_offset_y_px_active", info.get("pivot_offset_y_px", 0.0)))

    dx_arr = np.full_like(pivot_x_roi, dx_const, dtype=float)
    dy_arr = np.full_like(pivot_x_roi, dy_const, dtype=float)
    return dx_arr, dy_arr, "constant_offset"


def compute_px_per_in(cfg, raw, source="avg"):
    length_info = cfg.get("length_info", {})
    mode = length_info.get("mode", "")

    if mode in ("marker_diameter", "marker_diameter_multiframe"):
        if source == "green":
            px_per_in = length_info.get("px_per_in_green", None)
        elif source == "pink":
            px_per_in = length_info.get("px_per_in_pink", None)
        elif source == "avg_markers":
            px_per_in = length_info.get("px_per_in_avg_markers", length_info.get("px_per_in_avg", None))
        else:
            px_per_in = length_info.get("px_per_in_avg", None)

        if px_per_in is None or px_per_in <= 0:
            raise ValueError(f"{mode} mode exists but px_per_in for source='{source}' is invalid.")

        return float(px_per_in), f"{mode}:{source}"

    elif mode == "rod_length":
        rod_length_in = float(length_info["rod_length_in"])

        pivot_x = raw["pivot_x_roi"].astype(float)
        pivot_y = raw["pivot_y_roi"].astype(float)
        bob_x = raw["bob_x_roi"].astype(float)
        bob_y = raw["bob_y_roi"].astype(float)
        pivot_found = raw["pivot_found"].astype(int)
        bob_found = raw["bob_found"].astype(int)

        dx_off, dy_off, _ = get_pivot_offset_arrays_px(cfg, pivot_x)

        pivot_x_corr = pivot_x + dx_off
        pivot_y_corr = pivot_y + dy_off

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

def soft_deadband(values, threshold):
    """
    Soft-threshold / shrinkage:
      y = sign(x) * max(|x| - threshold, 0)
    """
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)

    mask = np.isfinite(values)
    v = values[mask]
    out[mask] = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    return out
        
def apply_stop_deadband(time_s, pos_in, vel_in_s, command_end_time_s,
                        buffer_s=0.3, late_noise_t0=20.0,
                        sigma_mult=3.0, final_pos_window=101,
                        final_pos_tol_in=0.03):
    """
    After commanded stop, clamp small vision-derived velocities to zero
    using a noise-based threshold and a final-position tolerance.
    """
    time_s = np.asarray(time_s, dtype=float)
    pos_in = np.asarray(pos_in, dtype=float)
    vel_in_s = np.asarray(vel_in_s, dtype=float)

    vel_out = vel_in_s.copy()

    # estimate velocity noise floor from late-time region
    late_mask = np.isfinite(vel_in_s) & (time_s >= late_noise_t0)
    if np.sum(late_mask) >= 10:
        sigma_v = float(np.nanstd(vel_in_s[late_mask]))
    else:
        sigma_v = float(np.nanstd(vel_in_s[np.isfinite(vel_in_s)]))

    if not np.isfinite(sigma_v):
        sigma_v = 0.0

    v_thresh = sigma_mult * sigma_v

    # estimate final position as slow average
    pos_slow = moving_average_nan(pos_in, final_pos_window)
    final_mask = np.isfinite(pos_slow) & (time_s >= late_noise_t0)
    if np.sum(final_mask) >= 10:
        final_pos_mean = float(np.nanmean(pos_slow[final_mask]))
    else:
        final_pos_mean = float(np.nanmean(pos_slow[np.isfinite(pos_slow)]))

    # stop region
    stop_mask = time_s >= (command_end_time_s + buffer_s)

    # condition 1: velocity is within noise deadband
    small_v = np.abs(vel_out) <= v_thresh

    # condition 2: position is basically sitting at final value
    near_final_pos = np.isfinite(pos_slow) & np.isfinite(final_pos_mean) & (
        np.abs(pos_slow - final_pos_mean) <= final_pos_tol_in
    )

    # soft deadband version
    vel_soft = soft_deadband(vel_out, v_thresh)
    
    # gradual fade after stop
    fade_start = command_end_time_s + buffer_s
    fade_duration_s =2.0   # try 1.5 to 3.0 s
    
    alpha = np.zeros_like(time_s, dtype=float)
    fade_mask = time_s >= fade_start
    alpha[fade_mask] = np.clip((time_s[fade_mask] - fade_start) / fade_duration_s, 0.0, 1.0)
    
    # blend original velocity with soft-deadbanded velocity
    vel_blend = (1.0 - alpha) * vel_out + alpha * vel_soft
    
    # optional stronger zeroing only when clearly settled
    very_small_v = np.abs(vel_blend) <= (0.35 * v_thresh)
    hard_zero_mask = stop_mask & near_final_pos & very_small_v
    vel_blend[hard_zero_mask] = 0.0
    
    vel_out = vel_blend

    return vel_out, {
        "sigma_v": sigma_v,
        "v_thresh": v_thresh,
        "final_pos_mean": final_pos_mean
    }

def parse_condition_and_trial(path_or_name):
    """
    Example:
      H_OL_21IN_T1 -> condition='H_OL_21IN_T', trial=1
    """
    stem = os.path.splitext(os.path.basename(path_or_name))[0]
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2)), stem
    return stem, np.nan, stem


def append_metrics_row(csv_path, row_dict):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def detect_free_decay_start(
    time_s,
    vel_in_s,
    pos_in,
    command_end_time_s,
    min_search_delay_s=0.0,
    vel_sigma_mult=3.0,
    min_vel_thresh_in_s=0.05,
    dwell_s=0.25,
    final_pos_window=101,
    final_pos_tol_in=0.03
):
    """
    Detect free-decay start as the first time after command_end_time_s where:
      - |velocity| is small
      - position is close to final value
      - those conditions persist for a short dwell time
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(vel_in_s, dtype=float)
    x = np.asarray(pos_in, dtype=float)

    dt = float(np.median(np.diff(t))) if len(t) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return command_end_time_s, {
            "vel_thresh_in_s": np.nan,
            "final_pos_mean_in": np.nan,
            "method": "fallback_command_end"
        }

    start_time = command_end_time_s + min_search_delay_s
    dwell_n = max(1, int(round(dwell_s / dt)))

    # Estimate noise floor from the tail of the record
    tail_t0 = max(start_time + 1.0, t[-1] - 5.0)
    tail_mask = np.isfinite(v) & (t >= tail_t0)
    if np.sum(tail_mask) >= 10:
        sigma_v = float(np.nanstd(v[tail_mask]))
    else:
        sigma_v = float(np.nanstd(v[np.isfinite(v)]))

    if not np.isfinite(sigma_v):
        sigma_v = 0.0

    vel_thresh = max(min_vel_thresh_in_s, vel_sigma_mult * sigma_v)

    # Final position estimate
    x_slow = moving_average_nan(x, final_pos_window)
    tail_x_mask = np.isfinite(x_slow) & (t >= tail_t0)
    if np.sum(tail_x_mask) >= 10:
        x_final = float(np.nanmean(x_slow[tail_x_mask]))
    else:
        x_final = float(np.nanmean(x_slow[np.isfinite(x_slow)]))

    good = (
        np.isfinite(v)
        & np.isfinite(x_slow)
        & (t >= start_time)
        & (np.abs(v) <= vel_thresh)
        & (np.abs(x_slow - x_final) <= final_pos_tol_in)
    )

    good_idx = np.where(good)[0]
    if good_idx.size == 0:
        return command_end_time_s, {
            "vel_thresh_in_s": vel_thresh,
            "final_pos_mean_in": x_final,
            "method": "fallback_command_end"
        }

    run_start = good_idx[0]
    run_len = 1

    for k in range(1, good_idx.size):
        if good_idx[k] == good_idx[k - 1] + 1:
            run_len += 1
        else:
            run_start = good_idx[k]
            run_len = 1

        if run_len >= dwell_n:
            idx0 = good_idx[k - run_len + 1]
            return float(t[idx0]), {
                "vel_thresh_in_s": vel_thresh,
                "final_pos_mean_in": x_final,
                "method": "velocity_position_dwell"
            }

    return command_end_time_s, {
        "vel_thresh_in_s": vel_thresh,
        "final_pos_mean_in": x_final,
        "method": "fallback_command_end"
    }

def detect_move_end_from_position(
    time_s,
    pos_in,
    vel_in_s,
    onset_frac=0.05,
    onset_min_in=0.15,
    vel_sigma_mult=3.0,
    min_vel_thresh_in_s=0.05,
    dwell_s=0.25,
    initial_pos_window=101,
    final_pos_window=101,
    final_pos_tol_in=0.03
):
    """
    Detect the end of commanded motion directly from position + velocity,
    without relying on a hard-coded command_end_time_s.

    Returns:
      move_end_s
      info dict
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(pos_in, dtype=float)
    v = np.asarray(vel_in_s, dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid) < 5:
        return np.nan, {"method": "insufficient_data"}

    dt = float(np.median(np.diff(t[np.isfinite(t)]))) if np.sum(np.isfinite(t)) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, {"method": "bad_dt"}

    x_slow = moving_average_nan(x, final_pos_window)

    # Initial and final position estimates
    x_init = mean_first_n_valid(x_slow, np.isfinite(x_slow), initial_pos_window)
    x_final = mean_last_n_valid(x_slow, np.isfinite(x_slow), final_pos_window)

    if not np.isfinite(x_init) or not np.isfinite(x_final):
        return np.nan, {"method": "bad_position_estimates"}

    total_travel = abs(x_final - x_init)
    onset_thresh = max(onset_min_in, onset_frac * total_travel)

    # Motion onset: first time position clearly leaves initial position
    onset_mask = np.isfinite(x_slow) & (np.abs(x_slow - x_init) >= onset_thresh)
    onset_idx = np.where(onset_mask)[0]
    if onset_idx.size == 0:
        return np.nan, {
            "method": "no_motion_onset_found",
            "x_init": x_init,
            "x_final": x_final,
            "total_travel": total_travel
        }

    motion_start_i = int(onset_idx[0])

    # Velocity noise floor from late part of record
    tail_t0 = max(t[motion_start_i] + 1.0, t[-1] - 5.0)
    tail_mask = np.isfinite(v) & (t >= tail_t0)
    if np.sum(tail_mask) >= 10:
        sigma_v = float(np.nanstd(v[tail_mask]))
    else:
        sigma_v = float(np.nanstd(v[np.isfinite(v)]))

    if not np.isfinite(sigma_v):
        sigma_v = 0.0

    vel_thresh = max(min_vel_thresh_in_s, vel_sigma_mult * sigma_v)
    dwell_n = max(1, int(round(dwell_s / dt)))

    # "steady at final position" condition
    good = (
        np.isfinite(v)
        & np.isfinite(x_slow)
        & (np.arange(len(t)) >= motion_start_i)
        & (np.abs(v) <= vel_thresh)
        & (np.abs(x_slow - x_final) <= final_pos_tol_in)
    )

    good_idx = np.where(good)[0]
    if good_idx.size == 0:
        return np.nan, {
            "method": "no_move_end_found",
            "x_init": x_init,
            "x_final": x_final,
            "total_travel": total_travel,
            "vel_thresh_in_s": vel_thresh
        }

    run_len = 1
    for k in range(1, good_idx.size):
        if good_idx[k] == good_idx[k - 1] + 1:
            run_len += 1
        else:
            run_len = 1

        if run_len >= dwell_n:
            idx0 = good_idx[k - run_len + 1]
            return float(t[idx0]), {
                "method": "position_and_velocity_dwell",
                "motion_start_s": float(t[motion_start_i]),
                "x_init": x_init,
                "x_final": x_final,
                "total_travel": total_travel,
                "vel_thresh_in_s": vel_thresh
            }

    return np.nan, {
        "method": "no_dwell_run_found",
        "motion_start_s": float(t[motion_start_i]),
        "x_init": x_init,
        "x_final": x_final,
        "total_travel": total_travel,
        "vel_thresh_in_s": vel_thresh
    }


def find_band_settling_time(time_s, theta_deg, start_time_s, band_deg):
    """
    Direct measured settling time to ±band_deg.
    Returns the first time after start_time_s such that all remaining samples
    stay within the band. Returns nan if the signal never settles in-record.
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)
    abs_th = np.abs(th)

    valid = np.isfinite(abs_th)
    if np.sum(valid) == 0:
        return np.nan

    tail_max = np.full_like(abs_th, np.nan, dtype=float)
    running = -np.inf

    for i in range(len(abs_th) - 1, -1, -1):
        if np.isfinite(abs_th[i]):
            running = max(running, abs_th[i])
        tail_max[i] = running if np.isfinite(running) else np.nan

    candidates = np.where(
        np.isfinite(tail_max)
        & (t >= start_time_s)
        & (tail_max <= band_deg)
    )[0]

    if candidates.size == 0:
        return np.nan

    return float(t[candidates[0]])


def extract_free_decay_metrics(
    time_s,
    theta_deg,
    free_decay_start_s,
    command_end_time_s,
    theta_band_deg,
    min_peak_deg=0.5,
    min_peak_prom_deg=0.15,
    min_peak_spacing_s=0.15
):
    """
    Extract peak sway, Td, fd, wd, zeta, wn, and settling metrics
    from the free-decay portion of theta(t).
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    use = np.isfinite(t) & np.isfinite(th) & (t >= free_decay_start_s)
    if np.sum(use) < 5:
        return {
            "free_decay_start_s": free_decay_start_s,
            "first_peak_abs_deg": np.nan,
            "max_poststop_abs_deg": np.nan,
            "n_abs_peaks": 0,
            "n_same_phase_peaks": 0,
            "Td_s": np.nan,
            "fd_hz": np.nan,
            "wd_rad_s": np.nan,
            "zeta_logdec": np.nan,
            "wn_rad_s": np.nan,
            "t_settle_band_s_measured": np.nan,
            "t_settle_2pct_s_est": np.nan
        }

    t_fd = t[use]
    th_fd = th[use]
    abs_fd = np.abs(th_fd)

    dt = float(np.median(np.diff(t_fd))) if len(t_fd) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 1e-3

    min_distance_samples = max(1, int(round(min_peak_spacing_s / dt)))

    peaks_abs, props = find_peaks(
        abs_fd,
        height=min_peak_deg,
        prominence=min_peak_prom_deg,
        distance=min_distance_samples
    )

    first_peak_abs_deg = float(abs_fd[peaks_abs[0]]) if len(peaks_abs) >= 1 else np.nan
    max_poststop_abs_deg = float(np.nanmax(abs_fd)) if np.any(np.isfinite(abs_fd)) else np.nan

    # Use every other abs peak so we compare approximately same-phase peaks
    same_phase = peaks_abs[::2]

    if len(same_phase) >= 2:
        same_t = t_fd[same_phase]
        same_a = abs_fd[same_phase]

        Td_vals = np.diff(same_t)
        Td_s = float(np.nanmean(Td_vals)) if Td_vals.size > 0 else np.nan
        fd_hz = float(1.0 / Td_s) if np.isfinite(Td_s) and Td_s > 0 else np.nan
        wd_rad_s = float(2.0 * np.pi / Td_s) if np.isfinite(Td_s) and Td_s > 0 else np.nan

        deltas = np.log(same_a[:-1] / same_a[1:])
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]

        if deltas.size > 0:
            delta_mean = float(np.nanmean(deltas))
            zeta = float(delta_mean / np.sqrt((2.0 * np.pi) ** 2 + delta_mean ** 2))
        else:
            zeta = np.nan

        if np.isfinite(zeta) and np.isfinite(wd_rad_s) and (zeta < 1.0):
            wn_rad_s = float(wd_rad_s / np.sqrt(1.0 - zeta ** 2))
        else:
            wn_rad_s = np.nan
    else:
        Td_s = np.nan
        fd_hz = np.nan
        wd_rad_s = np.nan
        zeta = np.nan
        wn_rad_s = np.nan

    t_settle_band = find_band_settling_time(t, th, free_decay_start_s, theta_band_deg)

    if np.isfinite(zeta) and np.isfinite(wn_rad_s) and zeta > 0 and wn_rad_s > 0:
        t_settle_2pct_est = float(4.0 / (zeta * wn_rad_s))
    else:
        t_settle_2pct_est = np.nan

    return {
        "free_decay_start_s": float(free_decay_start_s),
        "first_peak_abs_deg": first_peak_abs_deg,
        "max_poststop_abs_deg": max_poststop_abs_deg,
        "n_abs_peaks": int(len(peaks_abs)),
        "n_same_phase_peaks": int(len(same_phase)),
        "Td_s": Td_s,
        "fd_hz": fd_hz,
        "wd_rad_s": wd_rad_s,
        "zeta_logdec": zeta,
        "wn_rad_s": wn_rad_s,
        "t_settle_band_s_measured": t_settle_band,
        "t_settle_2pct_s_est": t_settle_2pct_est
    }

def measured_band_settling_with_censor(time_s, theta_deg, start_time_s, band_deg):
    """
    Returns:
      settle_time_s: first time after start_time_s that signal stays within ±band_deg
      settled_observed: True if observed in-record, False if not
      censor_lower_bound_s: lower bound if not observed, measured from start_time_s
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    valid = np.isfinite(t) & np.isfinite(th)
    if np.sum(valid) == 0:
        return np.nan, False, np.nan

    t = t[valid]
    th = th[valid]
    abs_th = np.abs(th)

    tail_max = np.full_like(abs_th, np.nan, dtype=float)
    running = -np.inf

    for i in range(len(abs_th) - 1, -1, -1):
        running = max(running, abs_th[i])
        tail_max[i] = running

    candidates = np.where((t >= start_time_s) & (tail_max <= band_deg))[0]

    if candidates.size > 0:
        return float(t[candidates[0]]), True, np.nan

    return np.nan, False, float(t[-1] - start_time_s)

def detect_move_start_from_position(
    time_s,
    pos_in,
    vel_in_s,
    onset_frac=0.03,
    onset_min_in=0.10,
    vel_sigma_mult=3.0,
    min_vel_thresh_in_s=0.05,
    dwell_s=0.15,
    initial_pos_window=101,
    final_pos_window=101
):
    """
    Detect trolley motion onset from position + velocity.

    Returns:
      move_start_s
      info dict
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(pos_in, dtype=float)
    v = np.asarray(vel_in_s, dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid) < 5:
        return np.nan, {"method": "insufficient_data"}

    dt = float(np.median(np.diff(t[np.isfinite(t)]))) if np.sum(np.isfinite(t)) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, {"method": "bad_dt"}

    x_slow = moving_average_nan(x, initial_pos_window)

    x_init = mean_first_n_valid(x_slow, np.isfinite(x_slow), initial_pos_window)
    x_final = mean_last_n_valid(x_slow, np.isfinite(x_slow), final_pos_window)

    if not np.isfinite(x_init) or not np.isfinite(x_final):
        return np.nan, {"method": "bad_position_estimates"}

    total_travel = abs(x_final - x_init)
    pos_thresh = max(onset_min_in, onset_frac * total_travel)

    # early noise floor for velocity
    early_t1 = min(t[-1], t[0] + 5.0)
    early_mask = np.isfinite(v) & (t <= early_t1)
    if np.sum(early_mask) >= 10:
        sigma_v = float(np.nanstd(v[early_mask]))
    else:
        sigma_v = float(np.nanstd(v[np.isfinite(v)]))

    if not np.isfinite(sigma_v):
        sigma_v = 0.0

    vel_thresh = max(min_vel_thresh_in_s, vel_sigma_mult * sigma_v)
    dwell_n = max(1, int(round(dwell_s / dt)))

    moved_from_init = np.isfinite(x_slow) & (np.abs(x_slow - x_init) >= pos_thresh)
    moving_fast = np.isfinite(v) & (np.abs(v) >= vel_thresh)

    good = moved_from_init | moving_fast
    good_idx = np.where(good)[0]

    if good_idx.size == 0:
        return np.nan, {
            "method": "no_motion_onset_found",
            "x_init": x_init,
            "x_final": x_final,
            "total_travel": total_travel,
            "pos_thresh_in": pos_thresh,
            "vel_thresh_in_s": vel_thresh
        }

    run_len = 1
    for k in range(1, good_idx.size):
        if good_idx[k] == good_idx[k - 1] + 1:
            run_len += 1
        else:
            run_len = 1

        if run_len >= dwell_n:
            idx0 = good_idx[k - run_len + 1]
            return float(t[idx0]), {
                "method": "position_or_velocity_dwell",
                "x_init": x_init,
                "x_final": x_final,
                "total_travel": total_travel,
                "pos_thresh_in": pos_thresh,
                "vel_thresh_in_s": vel_thresh
            }

    return float(t[good_idx[0]]), {
        "method": "first_trigger_only",
        "x_init": x_init,
        "x_final": x_final,
        "total_travel": total_travel,
        "pos_thresh_in": pos_thresh,
        "vel_thresh_in_s": vel_thresh
    }
def settling_time_from_fraction_of_ref(
    time_s,
    theta_deg,
    start_time_s,
    ref_amp_deg,
    frac=0.02,
    require_sustained=True
):
    """
    Settling time based on a fraction of a reference amplitude.

    Parameters
    ----------
    time_s : array
    theta_deg : array
    start_time_s : float
        Time after which settling is evaluated.
    ref_amp_deg : float
        Reference amplitude, e.g. max_abs_theta_deg or max_poststop_abs_deg.
    frac : float
        Fraction of ref amplitude, e.g. 0.02 for 2%.
    require_sustained : bool
        If True, returns first time after which all remaining samples stay
        within the threshold. If False, returns first crossing only.

    Returns
    -------
    settle_time_s : float
    threshold_deg : float
    observed : bool
    censor_lower_bound_s : float
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    valid = np.isfinite(t) & np.isfinite(th)
    if np.sum(valid) == 0 or not np.isfinite(ref_amp_deg) or ref_amp_deg <= 0:
        return np.nan, np.nan, False, np.nan

    t = t[valid]
    th = th[valid]
    abs_th = np.abs(th)

    threshold_deg = float(frac * ref_amp_deg)

    mask = t >= start_time_s
    if not np.any(mask):
        return np.nan, threshold_deg, False, np.nan

    if require_sustained:
        tail_max = np.full_like(abs_th, np.nan, dtype=float)
        running = -np.inf
        for i in range(len(abs_th) - 1, -1, -1):
            running = max(running, abs_th[i])
            tail_max[i] = running

        candidates = np.where(mask & (tail_max <= threshold_deg))[0]
    else:
        candidates = np.where(mask & (abs_th <= threshold_deg))[0]

    if candidates.size > 0:
        return float(t[candidates[0]]), threshold_deg, True, np.nan

    return np.nan, threshold_deg, False, float(t[-1] - start_time_s)

def detect_motion_window_from_plateaus(
    time_s,
    pos_in,
    vel_in_s=None,
    pos_window=101,
    vel_thresh_in_s=0.08,
    dwell_s=0.50,
    final_pos_tol_in=0.10
):
    """
    Detect motion start/end from flat position plateaus.

    Returns
    -------
    move_start_s, move_end_s, info
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(pos_in, dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid) < 5:
        return np.nan, np.nan, {"method": "insufficient_data"}

    t = t[valid]
    x = x[valid]

    x_slow = moving_average_nan(x, pos_window)

    if vel_in_s is None:
        v = gradient_with_time(x_slow, t)
    else:
        v_full = np.asarray(vel_in_s, dtype=float)
        v = v_full[valid]

    dt = float(np.median(np.diff(t))) if len(t) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan, {"method": "bad_dt"}

    dwell_n = max(2, int(round(dwell_s / dt)))

    x_init = mean_first_n_valid(x_slow, np.isfinite(x_slow), min(pos_window, len(x_slow)))
    x_final = mean_last_n_valid(x_slow, np.isfinite(x_slow), min(pos_window, len(x_slow)))

    if not np.isfinite(x_init) or not np.isfinite(x_final):
        return np.nan, np.nan, {"method": "bad_plateau_means"}

    # Initial flat plateau: low speed and near initial position
    initial_flat = (
        np.isfinite(v)
        & np.isfinite(x_slow)
        & (np.abs(v) <= vel_thresh_in_s)
        & (np.abs(x_slow - x_init) <= final_pos_tol_in)
    )

    # Final flat plateau: low speed and near final position
    final_flat = (
        np.isfinite(v)
        & np.isfinite(x_slow)
        & (np.abs(v) <= vel_thresh_in_s)
        & (np.abs(x_slow - x_final) <= final_pos_tol_in)
    )

    def first_run(mask, min_len):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        run_len = 1
        run_start = idx[0]
        for k in range(1, len(idx)):
            if idx[k] == idx[k - 1] + 1:
                run_len += 1
            else:
                run_start = idx[k]
                run_len = 1
            if run_len >= min_len:
                return (idx[k - run_len + 1], idx[k])
        if run_len >= min_len:
            return (idx[-run_len], idx[-1])
        return None

    def last_run(mask, min_len):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        runs = []
        run_start = idx[0]
        run_len = 1
        for k in range(1, len(idx)):
            if idx[k] == idx[k - 1] + 1:
                run_len += 1
            else:
                if run_len >= min_len:
                    runs.append((idx[k - run_len], idx[k - 1]))
                run_start = idx[k]
                run_len = 1
        if run_len >= min_len:
            runs.append((idx[-run_len], idx[-1]))
        return runs[-1] if runs else None

    init_run = first_run(initial_flat, dwell_n)
    final_run = last_run(final_flat, dwell_n)

    if init_run is None or final_run is None:
        return np.nan, np.nan, {
            "method": "plateau_not_found",
            "x_init": x_init,
            "x_final": x_final
        }

    # Motion starts when the initial plateau ends
    move_start_i = init_run[1] + 1

    # Motion ends when the final plateau begins
    move_end_i = final_run[0]

    if move_start_i >= len(t):
        move_start_i = len(t) - 1
    if move_end_i >= len(t):
        move_end_i = len(t) - 1

    return float(t[move_start_i]), float(t[move_end_i]), {
        "method": "plateau_bounds",
        "x_init": x_init,
        "x_final": x_final,
        "vel_thresh_in_s": vel_thresh_in_s,
        "dwell_s": dwell_s,
        "move_start_idx": int(move_start_i),
        "move_end_idx": int(move_end_i)
    }

def find_last_good_before_long_loss(valid_mask, n_bad=20):
    """
    Return the last reliable index before a long bad-data streak begins.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bad_run = 0

    for i, ok in enumerate(valid_mask):
        if ok:
            bad_run = 0
        else:
            bad_run += 1
            if bad_run >= n_bad:
                return i - bad_run

    return len(valid_mask) - 1

def first_run_start(mask, min_len):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    run_len = 1
    for k in range(1, len(idx)):
        if idx[k] == idx[k - 1] + 1:
            run_len += 1
        else:
            run_len = 1

        if run_len >= min_len:
            return idx[k - run_len + 1]

    if run_len >= min_len:
        return idx[-run_len]

    return None


def detect_motion_window_from_flat_position(
    time_s,
    pos_in,
    smooth_window=101,
    pos_frac=0.03,
    pos_min_in=0.10,
    vel_thresh_in_s=0.10,
    dwell_s=0.50
):
    """
    Detect motion start/end from the initial flat plateau and final flat plateau.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(pos_in, dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid) < 10:
        return np.nan, np.nan, {"method": "insufficient_data"}

    t = t[valid]
    x = x[valid]

    x_interp = interpolate_nans(t, x)
    x_slow = moving_average_nan(x_interp, smooth_window)
    v_slow = gradient_with_time(x_slow, t)

    dt = float(np.median(np.diff(t)))
    dwell_n = max(2, int(round(dwell_s / dt)))

    # Use short edge plateaus so the initial/final estimates are not polluted
    # by motion. The old len/4 rule could include active travel in short videos.
    edge_window_s = 0.50
    n_edge_time = int(round(edge_window_s / dt)) if np.isfinite(dt) and dt > 0 else smooth_window
    n_edge = max(5, min(smooth_window, n_edge_time, max(5, len(x_slow) // 10)))
    n_edge = min(n_edge, max(5, len(x_slow) // 4))

    x_init = float(np.nanmedian(x_slow[:n_edge]))
    x_final = float(np.nanmedian(x_slow[-n_edge:]))

    total_travel = abs(x_final - x_init)
    pos_thresh = max(pos_min_in, pos_frac * total_travel)

    # motion starts when we leave the initial plateau
    away_from_init = np.abs(x_slow - x_init) >= pos_thresh
    move_start_i = first_run_start(away_from_init, dwell_n)

    if move_start_i is None:
        return np.nan, np.nan, {
            "method": "no_motion_start_found",
            "x_init": x_init,
            "x_final": x_final,
            "total_travel": total_travel
        }

    # motion ends when we reach the final plateau and velocity is small
    near_final = np.abs(x_slow - x_final) <= pos_thresh
    final_flat = near_final

    # only search for end after motion has already started
    final_flat[:move_start_i + dwell_n] = False
    move_end_i = first_run_start(final_flat, dwell_n)

    if move_end_i is None:
        return float(t[move_start_i]), np.nan, {
            "method": "no_motion_end_found",
            "x_init": x_init,
            "x_final": x_final,
            "total_travel": total_travel,
            "pos_thresh_in": pos_thresh,
            "vel_thresh_in_s": vel_thresh_in_s
        }

    return float(t[move_start_i]), float(t[move_end_i]), {
        "method": "flat_plateau_bounds",
        "x_init": x_init,
        "x_final": x_final,
        "total_travel": total_travel,
        "pos_thresh_in": pos_thresh,
        "vel_thresh_in_s": vel_thresh_in_s
    }


# ============================================================
# Stationary-noise, placement, and deadband-settling helpers
# ============================================================

def parse_bool(value, default=False):
    """Parse booleans safely from config values that may be bools or strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in ("true", "t", "yes", "y", "1", "on"):
            return True
        if txt in ("false", "f", "no", "n", "0", "off"):
            return False
    return default


def json_safe(value):
    """Convert numpy scalars / NaN values to JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_stationary_noise_profile(profile_path):
    if not profile_path or not os.path.isfile(profile_path):
        return None
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stationary_noise_profile(profile_path, profile):
    folder = os.path.dirname(os.path.abspath(profile_path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(profile), f, indent=4)


def build_stationary_noise_profile(time_s, theta_deg, vx_in_s, x_in, theta_valid, pivot_valid, meta=None):
    """
    Save scalar noise statistics from a no-motion run.
    This intentionally does NOT save a waveform for subtraction.
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)
    vx = np.asarray(vx_in_s, dtype=float)
    x = np.asarray(x_in, dtype=float)

    theta_mask = np.asarray(theta_valid, dtype=bool) & np.isfinite(th)
    vx_mask = np.isfinite(vx)
    x_mask = np.asarray(pivot_valid, dtype=bool) & np.isfinite(x)

    theta_vals = th[theta_mask]
    vx_vals = vx[vx_mask]
    x_vals = x[x_mask]

    if x_vals.size > 0:
        x_detrended = x_vals - float(np.nanmean(x_vals))
    else:
        x_detrended = x_vals

    profile = {
        "profile_type": "stationary_noise_profile",
        "format_version": 1,
        "created_by": "analyze_data_windows_S2_updated.py",
        "n_samples_total": int(len(t)),
        "n_theta_valid": int(theta_vals.size),
        "n_vx_valid": int(vx_vals.size),
        "n_x_valid": int(x_vals.size),
        "time_start_s": float(np.nanmin(t)) if np.any(np.isfinite(t)) else np.nan,
        "time_end_s": float(np.nanmax(t)) if np.any(np.isfinite(t)) else np.nan,
        "theta_mean_deg": float(np.nanmean(theta_vals)) if theta_vals.size else np.nan,
        "theta_std_deg": float(np.nanstd(theta_vals, ddof=1)) if theta_vals.size > 1 else np.nan,
        "theta_maxabs_deg": float(np.nanmax(np.abs(theta_vals))) if theta_vals.size else np.nan,
        "theta_ptp_deg": float(np.nanmax(theta_vals) - np.nanmin(theta_vals)) if theta_vals.size else np.nan,
        "vx_std_in_s": float(np.nanstd(vx_vals, ddof=1)) if vx_vals.size > 1 else np.nan,
        "vx_maxabs_in_s": float(np.nanmax(np.abs(vx_vals))) if vx_vals.size else np.nan,
        "x_std_in": float(np.nanstd(x_detrended, ddof=1)) if x_detrended.size > 1 else np.nan,
        "notes": "Use theta_mean_deg only as a constant bias correction. Do not subtract any stationary waveform."
    }

    if meta:
        profile["meta"] = meta

    return profile


def empty_decay_metrics(free_decay_start_s=np.nan):
    return {
        "free_decay_start_s": free_decay_start_s,
        "first_peak_abs_deg": np.nan,
        "max_poststop_abs_deg": np.nan,
        "n_abs_peaks": 0,
        "n_same_phase_peaks": 0,
        "Td_s": np.nan,
        "fd_hz": np.nan,
        "wd_rad_s": np.nan,
        "zeta_logdec": np.nan,
        "wn_rad_s": np.nan,
        "t_settle_band_s_measured": np.nan,
        "t_settle_2pct_s_est": np.nan
    }


def measured_deadband_settling_with_dwell(time_s, theta_deg, start_time_s, band_deg=0.5, dwell_s=1.0):
    """
    Primary control-relevant settling metric.

    Returns the first absolute time after start_time_s where |theta| <= band_deg
    continuously for dwell_s. If not observed, returns a censoring lower bound
    measured from start_time_s to the end of the record.
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    valid = np.isfinite(t) & np.isfinite(th)
    if np.sum(valid) < 2 or not np.isfinite(start_time_s):
        return np.nan, False, np.nan

    t = t[valid]
    th = th[valid]

    after = t >= start_time_s
    if not np.any(after):
        return np.nan, False, np.nan

    dt = float(np.nanmedian(np.diff(t))) if len(t) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, False, np.nan

    dwell_n = max(1, int(round(float(dwell_s) / dt)))
    inside = after & (np.abs(th) <= float(band_deg))
    idx = np.where(inside)[0]

    if idx.size == 0:
        return np.nan, False, float(t[-1] - start_time_s)

    run_len = 1
    for k in range(1, len(idx)):
        if idx[k] == idx[k - 1] + 1:
            run_len += 1
        else:
            run_len = 1

        if run_len >= dwell_n:
            i0 = idx[k - run_len + 1]
            return float(t[i0]), True, np.nan

    if run_len >= dwell_n:
        i0 = idx[-run_len]
        return float(t[i0]), True, np.nan

    return np.nan, False, float(t[-1] - start_time_s)


def generate_idealized_position_trace(time_s, measured_x_in, move_start_s, move_end_s, command_speed_in_s=4.6):
    """
    Build the project-level ideal position trace:
      flat before move_start, linear at commanded speed during motion,
      flat after move_end.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(measured_x_in, dtype=float)
    ideal = np.full_like(t, np.nan, dtype=float)

    if not (np.isfinite(move_start_s) and np.isfinite(move_end_s)) or move_end_s <= move_start_s:
        return ideal, {
            "method": "not_computed_invalid_motion_window",
            "ideal_x0_in": np.nan,
            "ideal_target_in": np.nan,
            "ideal_direction_sign": np.nan,
            "command_speed_in_s": float(command_speed_in_s),
            "command_duration_s": np.nan
        }

    pre_mask = np.isfinite(x) & (t <= move_start_s)
    if np.sum(pre_mask) >= 3:
        x0 = float(np.nanmedian(x[pre_mask]))
    else:
        first_valid = np.where(np.isfinite(x))[0]
        x0 = float(x[first_valid[0]]) if first_valid.size else 0.0

    post_mask = np.isfinite(x) & (t >= move_end_s)
    if np.sum(post_mask) >= 3:
        x_final_measured = float(np.nanmedian(x[post_mask]))
    else:
        valid = np.where(np.isfinite(x))[0]
        x_final_measured = float(x[valid[-1]]) if valid.size else x0

    dir_sign = np.sign(x_final_measured - x0)
    if dir_sign == 0 or not np.isfinite(dir_sign):
        dir_sign = 1.0

    duration = float(move_end_s - move_start_s)
    travel_ideal = float(command_speed_in_s) * duration
    target = x0 + dir_sign * travel_ideal

    tau = np.clip(t - move_start_s, 0.0, duration)
    ideal = x0 + dir_sign * float(command_speed_in_s) * tau
    ideal[t >= move_end_s] = target
    ideal[t <= move_start_s] = x0

    return ideal, {
        "method": "linear_constant_speed_between_detected_motion_bounds",
        "ideal_x0_in": x0,
        "ideal_target_in": target,
        "ideal_direction_sign": float(dir_sign),
        "command_speed_in_s": float(command_speed_in_s),
        "command_duration_s": duration,
        "ideal_travel_in": travel_ideal
    }


def compute_placement_metrics(time_s, measured_x_in, ideal_x_in, move_end_s, goal_in=0.125, poststop_buffer_s=0.0):
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(measured_x_in, dtype=float)
    ideal = np.asarray(ideal_x_in, dtype=float)
    delta = x - ideal

    if not np.isfinite(move_end_s):
        return delta, {
            "placement_error_mean_poststop_in": np.nan,
            "placement_error_maxabs_poststop_in": np.nan,
            "placement_goal_in": float(goal_in),
            "placement_pass": np.nan,
            "placement_n_poststop_samples": 0,
            "placement_method": "not_computed_invalid_motion_end"
        }

    mask = np.isfinite(delta) & (t >= float(move_end_s) + float(poststop_buffer_s))
    vals = delta[mask]

    if vals.size == 0:
        return delta, {
            "placement_error_mean_poststop_in": np.nan,
            "placement_error_maxabs_poststop_in": np.nan,
            "placement_goal_in": float(goal_in),
            "placement_pass": np.nan,
            "placement_n_poststop_samples": 0,
            "placement_method": "no_valid_poststop_samples"
        }

    maxabs = float(np.nanmax(np.abs(vals)))
    meanp = float(np.nanmean(vals)) 
    return delta, {
        "placement_error_mean_poststop_in": float(np.nanmean(vals)),
        "placement_error_maxabs_poststop_in": maxabs,
        "placement_goal_in": float(goal_in),
        "placement_pass": int(abs(meanp) <= float(goal_in)),
        "placement_n_poststop_samples": int(vals.size),
        "placement_method": "tracked_minus_ideal_poststop"
    }


def compute_motion_velocity_metrics(
    time_s,
    x_smooth_in,
    vx_smooth_in_s,
    move_start_s,
    move_end_s
):
    """
    Compute mean trolley velocity during the detected motion window.

    Returns stable travel/duration velocity and velocity-signal statistics.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(x_smooth_in, dtype=float)
    v = np.asarray(vx_smooth_in_s, dtype=float)

    if not (
        np.isfinite(move_start_s)
        and np.isfinite(move_end_s)
        and move_end_s > move_start_s
    ):
        return {
            "move_duration_s": np.nan,
            "move_x_start_in": np.nan,
            "move_x_end_in": np.nan,
            "move_travel_in": np.nan,
            "mean_velocity_from_travel_in_s": np.nan,
            "mean_velocity_signal_in_s": np.nan,
            "median_velocity_signal_in_s": np.nan,
            "mean_abs_velocity_signal_in_s": np.nan,
            "velocity_signal_std_in_s": np.nan,
            "velocity_n_samples": 0
        }

    valid_x = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid_x) < 2:
        x_start = np.nan
        x_end = np.nan
    else:
        x_start = float(np.interp(move_start_s, t[valid_x], x[valid_x]))
        x_end = float(np.interp(move_end_s, t[valid_x], x[valid_x]))

    duration = float(move_end_s - move_start_s)
    travel = float(x_end - x_start) if np.isfinite(x_start) and np.isfinite(x_end) else np.nan
    mean_from_travel = travel / duration if np.isfinite(travel) and duration > 0 else np.nan

    move_mask = (
        np.isfinite(t)
        & np.isfinite(v)
        & (t >= move_start_s)
        & (t <= move_end_s)
    )

    vals = v[move_mask]

    return {
        "move_duration_s": duration,
        "move_x_start_in": x_start,
        "move_x_end_in": x_end,
        "move_travel_in": travel,
        "mean_velocity_from_travel_in_s": mean_from_travel,
        "mean_velocity_signal_in_s": float(np.nanmean(vals)) if vals.size else np.nan,
        "median_velocity_signal_in_s": float(np.nanmedian(vals)) if vals.size else np.nan,
        "mean_abs_velocity_signal_in_s": float(np.nanmean(np.abs(vals))) if vals.size else np.nan,
        "velocity_signal_std_in_s": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan,
        "velocity_n_samples": int(vals.size)
    }

def append_metrics_row(csv_path, row_dict):
    """
    Append a metrics row, upgrading the header safely if new columns were added.
    A backup is made before any header rewrite.
    """
    row_dict = {str(k): json_safe(v) for k, v in row_dict.items()}

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            writer.writeheader()
            writer.writerow(row_dict)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)

    new_fields = old_fields[:]
    for k in row_dict.keys():
        if k not in new_fields:
            new_fields.append(k)

    if new_fields != old_fields:
        backup_path = csv_path + ".bak_before_column_update"
        if not os.path.isfile(backup_path):
            shutil.copy2(csv_path, backup_path)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            for old_row in old_rows:
                writer.writerow(old_row)
            writer.writerow(row_dict)
    else:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields)
            writer.writerow(row_dict)


def plot_vline_if_finite(x, **kwargs):
    if np.isfinite(x):
        plt.axvline(x, **kwargs)
        
def envelope_settling_time_from_peaks(
    time_s,
    theta_deg,
    start_time_s,
    band_deg=0.5,
    min_peak_prom_deg=0.05,
    min_peak_spacing_s=0.5
):
    """
    Envelope-based settling for oscillatory sway.

    Finds peaks of |theta| after start_time_s and returns the first peak time
    after which all remaining envelope peaks stay <= band_deg.

    This avoids calling the system 'settled' during a temporary trough.
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    valid = np.isfinite(t) & np.isfinite(th) & (t >= start_time_s)
    if np.sum(valid) < 10:
        return np.nan, False, np.nan, {
            "method": "insufficient_poststop_data",
            "n_envelope_peaks": 0,
            "first_peak_deg": np.nan,
            "last_peak_deg": np.nan
        }

    t_post = t[valid]
    abs_post = np.abs(th[valid])

    dt = float(np.nanmedian(np.diff(t_post))) if len(t_post) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, False, np.nan, {
            "method": "bad_dt",
            "n_envelope_peaks": 0,
            "first_peak_deg": np.nan,
            "last_peak_deg": np.nan
        }

    peak_distance_n = max(1, int(round(min_peak_spacing_s / dt)))

    peaks, _ = find_peaks(
        abs_post,
        prominence=min_peak_prom_deg,
        distance=peak_distance_n
    )

    if peaks.size == 0:
        # If no peaks exceed prominence, check whether the whole tail is inside band.
        if np.nanmax(abs_post) <= band_deg:
            return float(t_post[0]), True, np.nan, {
                "method": "whole_tail_inside_band_no_peaks",
                "n_envelope_peaks": 0,
                "first_peak_deg": np.nan,
                "last_peak_deg": np.nan
            }

        return np.nan, False, float(t_post[-1] - start_time_s), {
            "method": "no_peaks_but_tail_exceeds_band",
            "n_envelope_peaks": 0,
            "first_peak_deg": np.nan,
            "last_peak_deg": np.nan
        }

    peak_t = t_post[peaks]
    peak_a = abs_post[peaks]

    # Tail maximum of peak amplitudes
    tail_peak_max = np.full_like(peak_a, np.nan, dtype=float)
    running = -np.inf
    for i in range(len(peak_a) - 1, -1, -1):
        running = max(running, peak_a[i])
        tail_peak_max[i] = running

    candidates = np.where(tail_peak_max <= band_deg)[0]

    info = {
        "method": "envelope_peak_tail_check",
        "n_envelope_peaks": int(peaks.size),
        "first_peak_deg": float(peak_a[0]),
        "last_peak_deg": float(peak_a[-1]),
        "max_envelope_peak_deg": float(np.nanmax(peak_a))
    }

    if candidates.size > 0:
        settle_s = float(peak_t[candidates[0]])
        return settle_s, True, np.nan, info

    return np.nan, False, float(t_post[-1] - start_time_s), info

def smooth_theta_for_metrics(time_s, theta_deg, window_s=0.15, polyorder=2):
    """
    Lightly smooth theta only for settling/envelope metrics.

    This should not replace the plotted raw-corrected theta. It is only used
    to reduce pixel-scale jitter when evaluating small deadbands like ±0.5 deg.
    """
    t = np.asarray(time_s, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    valid = np.isfinite(t) & np.isfinite(th)
    if np.sum(valid) < 5:
        return th.copy(), {
            "method": "not_smoothed_insufficient_data",
            "window_samples": 0
        }

    th_interp = interpolate_nans(t, th)

    dt = float(np.nanmedian(np.diff(t[valid]))) if np.sum(valid) >= 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return th.copy(), {
            "method": "not_smoothed_bad_dt",
            "window_samples": 0
        }

    desired_window = int(round(window_s / dt))
    sg_window = make_valid_savgol_window(len(th_interp), desired_window, polyorder)

    if sg_window is None:
        return th.copy(), {
            "method": "not_smoothed_bad_window",
            "window_samples": 0
        }

    th_smooth = savgol_filter(
        th_interp,
        window_length=sg_window,
        polyorder=polyorder,
        deriv=0,
        delta=dt,
        mode="interp"
    )

    # preserve NaNs where original theta was invalid
    th_smooth[~np.isfinite(th)] = np.nan

    return th_smooth, {
        "method": "savgol_metric_theta_only",
        "window_s": float(window_s),
        "window_samples": int(sg_window),
        "polyorder": int(polyorder)
    }


# ============================================================
# Main analysis
# ============================================================

def main():
    
    print("\n=== Dual Marker Analysis — stationary noise + placement update ===\n")

    config_path = input('Enter config JSON path [default: config_dual_marker.json]: ').strip().strip('"')
    if not config_path:
        config_path = "config_dual_marker.json"

    default_raw_csv = make_tagged_name_from_trial_stem(config_path, "_RAW")
    default_out_csv = make_tagged_name_from_trial_stem(config_path, "_OUT")
    default_angle_plot = make_tagged_name_from_trial_stem(config_path, "_AP")
    default_pos_plot = make_tagged_name_from_trial_stem(config_path, "_PP")
    default_speed_plot = make_tagged_name_from_trial_stem(config_path, "_SPD")

    print("\nDerived default filenames:")
    print(f"  RAW : {default_raw_csv}")
    print(f"  OUT : {default_out_csv}")
    print(f"  AP  : {default_angle_plot}")
    print(f"  PP  : {default_pos_plot}")
    print(f"  SPD : {default_speed_plot}")

    raw_csv_path = input(f'Raw CSV path [default: {default_raw_csv}]: ').strip().strip('"') or default_raw_csv
    out_csv = input(f'Analyzed output CSV [default: {default_out_csv}]: ').strip().strip('"') or default_out_csv
    angle_plot_path = default_angle_plot
    pos_plot_path = default_pos_plot
    speed_plot_path = default_speed_plot

    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found:\n{config_path}")
        return

    if not os.path.isfile(raw_csv_path):
        print(f"ERROR: Raw CSV not found:\n{raw_csv_path}")
        return

    cfg = load_json(config_path)
    raw = load_csv_with_header(raw_csv_path)
    analysis_defaults = cfg.get("analysis_defaults", {})

    theta_band_deg = float(analysis_defaults.get("theta_band_deg", 0.5))
    settling_deadband_dwell_s = float(analysis_defaults.get("settling_deadband_dwell_s", 1.0))
    command_speed_in_s = float(analysis_defaults.get("command_speed_in_s", 4.6))
    placement_goal_in = float(analysis_defaults.get("placement_goal_in", 0.125))
    placement_poststop_buffer_s = float(analysis_defaults.get("placement_poststop_buffer_s", 0.0))
    no_motion_travel_threshold_in = float(analysis_defaults.get("no_motion_travel_threshold_in", 0.25))

    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    default_noise_profile_path = analysis_defaults.get(
        "stationary_noise_profile_path",
        os.path.join(config_dir, "stationary_noise_profile.json")
    )
    if not os.path.isabs(default_noise_profile_path):
        default_noise_profile_path = os.path.join(config_dir, default_noise_profile_path)

    #stationary_txt = input("Analyze this run as a stationary noise/reference run? [y/N]: ").strip().lower()
    #is_stationary_noise_run = stationary_txt == "y"

    #########################################################
    # EDIT HERE STAT FILE
    #########################################################
    
    # --------------------------------------------------------
    # Stationary noise profile setup
    # --------------------------------------------------------
    # Stationary runs are complete, so all future runs are treated as regular trials.
    # The stationary profile path is hardcoded here so you do not have to paste it every time.
    
    is_stationary_noise_run = False
    
    stationary_profile = None
    stationary_profile_path_used = ""
    stationary_theta_mean_subtracted_deg = 0.0
    
    # Manually set stationary noise profile path here:
    stationary_profile_path_used = r"C:/Users/David/OneDrive/Desktop/OpenCV/OpenCV Test/stationary_noise_profile.json"
    
    stationary_profile = load_stationary_noise_profile(stationary_profile_path_used)
    
    if stationary_profile is not None:
        stationary_theta_mean_subtracted_deg = float(stationary_profile.get("theta_mean_deg") or 0.0)
        print("\nLoaded stationary noise profile:")
        print(f"  {stationary_profile_path_used}")
        print(f"  theta_mean_deg subtracted: {stationary_theta_mean_subtracted_deg:+.6f}")
    else:
        print("\nWARNING: Stationary noise profile was not found or could not be loaded.")
        print(f"  Expected path: {stationary_profile_path_used}")
        print("  Continuing without stationary theta correction.")

    show_plots_txt = input("Show plots in Spyder/UI? [y/N]: ").strip().lower()
    show_plots = (show_plots_txt == "y")

    print("\nUsing filenames:")
    print(f"  config : {config_path}")
    print(f"  raw    : {raw_csv_path}")
    print(f"  out    : {out_csv}")
    print(f"  angle  : {angle_plot_path}")
    print(f"  pos    : {pos_plot_path}")
    print(f"  speed  : {speed_plot_path}")

    frame_idx = raw["frame_idx"].astype(int)
    time_s = raw["time_s"].astype(float)
    pivot_x = raw["pivot_x_roi"].astype(float)
    pivot_y = raw["pivot_y_roi"].astype(float)
    bob_x = raw["bob_x_roi"].astype(float)
    bob_y = raw["bob_y_roi"].astype(float)
    pivot_found = raw["pivot_found"].astype(int)
    bob_found = raw["bob_found"].astype(int)

    pivot_valid = (pivot_found == 1) & np.isfinite(pivot_x) & np.isfinite(pivot_y)
    theta_valid = ((pivot_found == 1) & (bob_found == 1) &
                   np.isfinite(pivot_x) & np.isfinite(pivot_y) &
                   np.isfinite(bob_x) & np.isfinite(bob_y))

    pivot_dx_px, pivot_dy_px, pivot_offset_mode_used = get_pivot_offset_arrays_px(cfg, pivot_x)
    pivot_x_corr = pivot_x + pivot_dx_px
    pivot_y_corr = pivot_y + pivot_dy_px
    
    # ============================================================
    # LINEAR DRIFT CORRECTION FOR pivot_y_roi
    # ============================================================
    # Correct for the measured linear drift in the pivot y pixel position.
    # This removes a systematic parallax/mechanical error from theta.
    # Only apply to frames where pivot is valid.
    
    use_pivot_y_drift_correction = True  # set False to disable
    
    if use_pivot_y_drift_correction:
        pivot_y_drift_mask = pivot_valid & np.isfinite(time_s) & np.isfinite(pivot_y_corr)
    
        if np.sum(pivot_y_drift_mask) >= 10:
            # Fit a line to pivot_y_corr vs time
            drift_coef = np.polyfit(time_s[pivot_y_drift_mask], pivot_y_corr[pivot_y_drift_mask], 1)
            pivot_y_drift_trend = np.polyval(drift_coef, time_s)
    
            # Remove drift relative to the initial value so the zeroing is preserved
            pivot_y_drift_correction = pivot_y_drift_trend - pivot_y_drift_trend[0]
            pivot_y_corr_driftcorr = pivot_y_corr - pivot_y_drift_correction
    
            print("\n--- pivot_y drift correction ---")
            print(f"  drift slope    : {drift_coef[0]:.6f} px/s")
            print(f"  drift intercept: {drift_coef[1]:.6f} px")
            print(f"  total drift removed over record: {pivot_y_drift_correction[-1] - pivot_y_drift_correction[0]:.3f} px")
    
            # Replace pivot_y_corr with drift-corrected version for all downstream theta computation
            pivot_y_corr = pivot_y_corr_driftcorr
        else:
            print("WARNING: Not enough valid pivot_y samples for drift correction.")
    
    print("\nPivot offset diagnostic:")
    print(f"  pivot_offset_mode_used      : {pivot_offset_mode_used}")
    print(f"  pivot_dx_px min/max/range   : {np.nanmin(pivot_dx_px):+.3f}, {np.nanmax(pivot_dx_px):+.3f}, {np.nanmax(pivot_dx_px) - np.nanmin(pivot_dx_px):.3f}")
    print(f"  pivot_dy_px min/max/range   : {np.nanmin(pivot_dy_px):+.3f}, {np.nanmax(pivot_dy_px):+.3f}, {np.nanmax(pivot_dy_px) - np.nanmin(pivot_dy_px):.3f}")
    
    if (np.nanmax(pivot_dx_px) - np.nanmin(pivot_dx_px)) > 20:
        print("WARNING: pivot_dx_px changes by more than 20 px. This can create fake x-position jumps.")
    if (np.nanmax(pivot_dy_px) - np.nanmin(pivot_dy_px)) > 20:
        print("WARNING: pivot_dy_px changes by more than 20 px. This can create fake theta jumps.")

    pivot_scale_source = analysis_defaults.get("pivot_scale_source", "green")
    bob_scale_source = analysis_defaults.get("bob_scale_source", "pink")
    fallback_scale_source = analysis_defaults.get("fallback_scale_source", "avg")

    try:
        px_per_in_pivot, pivot_px_source = compute_px_per_in(cfg, raw, source=pivot_scale_source)
    except Exception as e:
        print(f"WARNING: pivot scale source '{pivot_scale_source}' failed ({e}); using fallback '{fallback_scale_source}'.")
        px_per_in_pivot, pivot_px_source = compute_px_per_in(cfg, raw, source=fallback_scale_source)
    try:
        px_per_in_bob, bob_px_source = compute_px_per_in(cfg, raw, source=bob_scale_source)
    except Exception as e:
        print(f"WARNING: bob scale source '{bob_scale_source}' failed ({e}); using fallback '{fallback_scale_source}'.")
        px_per_in_bob, bob_px_source = compute_px_per_in(cfg, raw, source=fallback_scale_source)

    print(f"\nUsing pivot px/in = {px_per_in_pivot:.6f} from mode: {pivot_px_source}")
    print(f"Using bob   px/in = {px_per_in_bob:.6f} from mode: {bob_px_source}")

    pivot_x_in_raw = pivot_x_corr / px_per_in_pivot
    pivot_y_in_raw = pivot_y_corr / px_per_in_pivot
    bob_x_in_raw = bob_x / px_per_in_bob
    bob_y_in_raw = bob_y / px_per_in_bob
    pivot_x_in_raw[~pivot_valid] = np.nan
    pivot_y_in_raw[~pivot_valid] = np.nan
    bob_x_in_raw[bob_found != 1] = np.nan
    bob_y_in_raw[bob_found != 1] = np.nan

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
    pivot_x_in_report_interp = interpolate_nans(time_s, pivot_x_in_report)

    velocity_smoothing_method = analysis_defaults.get("velocity_smoothing_method", "savgol")
    savgol_window = int(analysis_defaults.get("savgol_window", 41))
    savgol_polyorder = int(analysis_defaults.get("savgol_polyorder", 3))
    speed_post_smoothing_window = int(analysis_defaults.get("speed_post_smoothing_window", 0))
    pivot_vx_in_s_raw = gradient_with_time(pivot_x_in_report_interp, time_s)

    if velocity_smoothing_method == "savgol":
        dt_med = float(np.median(np.diff(time_s))) if len(time_s) >= 2 else np.nan
        sg_window = make_valid_savgol_window(len(pivot_x_in_report_interp), savgol_window, savgol_polyorder)
        if sg_window is None or not np.isfinite(dt_med) or dt_med <= 0:
            print("WARNING: Savitzky-Golay settings invalid; falling back to moving average.")
            pivot_x_in_report_smooth = moving_average_nan(pivot_x_in_report_interp, position_smoothing_window)
            pivot_vx_in_s_report = gradient_with_time(pivot_x_in_report_smooth, time_s)
            pivot_vx_in_s_report_smooth = moving_average_nan(pivot_vx_in_s_report, speed_smoothing_window)
            sg_window_used = None
            velocity_smoothing_method_used = "moving_average_fallback"
        else:
            pivot_x_in_report_smooth = savgol_filter(pivot_x_in_report_interp, window_length=sg_window,
                                                     polyorder=savgol_polyorder, deriv=0, delta=dt_med, mode="interp")
            pivot_vx_in_s_report = savgol_filter(pivot_x_in_report_interp, window_length=sg_window,
                                                 polyorder=savgol_polyorder, deriv=1, delta=dt_med, mode="interp")
            pivot_vx_in_s_report_smooth = (moving_average_nan(pivot_vx_in_s_report, speed_post_smoothing_window)
                                           if speed_post_smoothing_window > 1 else pivot_vx_in_s_report.copy())
            sg_window_used = sg_window
            velocity_smoothing_method_used = "savgol"
    else:
        pivot_x_in_report_smooth = moving_average_nan(pivot_x_in_report_interp, position_smoothing_window)
        pivot_vx_in_s_report = gradient_with_time(pivot_x_in_report_smooth, time_s)
        pivot_vx_in_s_report_smooth = moving_average_nan(pivot_vx_in_s_report, speed_smoothing_window)
        sg_window_used = None
        velocity_smoothing_method_used = "moving_average"

    use_stop_deadband = parse_bool(analysis_defaults.get("use_stop_deadband", True), default=True)
    stop_deadband_buffer_s = float(analysis_defaults.get("stop_deadband_buffer_s", 0.3))
    stop_deadband_sigma_mult = float(analysis_defaults.get("stop_deadband_sigma_mult", 3.0))
    stop_deadband_final_pos_tol_in = float(analysis_defaults.get("stop_deadband_final_pos_tol_in", 0.03))

    last_det_i = find_last_good_before_long_loss(pivot_valid, n_bad=20)
    if last_det_i < 0:
        last_det_i = len(time_s) - 1
    t_det = time_s[:last_det_i + 1]
    # Use smoothed position for motion-window detection.
    # This should match the orange position plot better.
    x_det = pivot_x_in_report_smooth[:last_det_i + 1]

    motion_info = {"method": "skipped_stationary_noise_run"}
    move_start_detected_s = np.nan
    move_end_detected_s = np.nan
    if not is_stationary_noise_run:
        move_start_detected_s, move_end_detected_s, motion_info = detect_motion_window_from_flat_position(
            time_s=t_det,
            pos_in=x_det,
            smooth_window=int(analysis_defaults.get("motion_window_smooth_window", 101)),
            pos_frac=float(analysis_defaults.get("motion_window_pos_frac", 0.03)),
            pos_min_in=float(analysis_defaults.get("motion_window_pos_min_in", 0.10)),
            vel_thresh_in_s=float(analysis_defaults.get("motion_window_vel_thresh_in_s", 0.10)),
            dwell_s=float(analysis_defaults.get("motion_window_dwell_s", 0.50))
        )
        
 
    
    

    travel_est_for_guard = motion_info.get("total_travel", np.nan)
    if not np.isfinite(travel_est_for_guard):
        x_guard_start = mean_first_n_valid(pivot_x_in_report, pivot_valid, direction_window_n_frames)
        x_guard_end = mean_last_n_valid(pivot_x_in_report, pivot_valid, direction_window_n_frames)
        travel_est_for_guard = abs(x_guard_end - x_guard_start) if np.isfinite(x_guard_start) and np.isfinite(x_guard_end) else np.nan

    no_motion_detected = bool(is_stationary_noise_run or
                              (np.isfinite(travel_est_for_guard) and travel_est_for_guard < no_motion_travel_threshold_in))

    if no_motion_detected:
        if is_stationary_noise_run:
            print("\nStationary-noise mode: motion-window, placement, and free-decay metrics will be skipped.")
        else:
            print(f"\nWARNING: no-motion guard triggered. Estimated travel = {travel_est_for_guard:.6f} in, threshold = {no_motion_travel_threshold_in:.6f} in.")
        move_start_detected_s = np.nan
        move_end_detected_s = np.nan
        motion_info["no_motion_guard"] = True
        motion_info["no_motion_travel_threshold_in"] = no_motion_travel_threshold_in
    else:
        if not np.isfinite(move_start_detected_s):
            print("WARNING: move_start_detected_s could not be found; using first clean time as fallback.")
            move_start_detected_s = float(t_det[0]) if len(t_det) else np.nan
            motion_info["move_start_fallback"] = "first_clean_time"
        if not np.isfinite(move_end_detected_s):
            print("WARNING: move_end_detected_s could not be found; using last clean time as fallback.")
            move_end_detected_s = float(t_det[-1]) if len(t_det) else np.nan
            motion_info["move_end_fallback"] = "last_clean_time"
     
            
     
        
    # --------------------------------------------------------
    # Motion-window variables!!!!!
    # --------------------------------------------------------
    # Keep the detected motion window for theta/free-decay/settling metrics.
    # Use a separate adjusted window only for position/placement accuracy.
    
    move_start_theta_s = move_start_detected_s
    move_end_theta_s = move_end_detected_s

    
    move_start_position_s = move_start_detected_s
    move_end_position_s = move_end_detected_s
    
    # Manual position-window adjustment.
    # This is intentionally only for idealized position and placement accuracy.
    use_manual_position_window_adjustment = True
    
    if use_manual_position_window_adjustment and not no_motion_detected:
        move_start_position_s = move_start_position_s + (0.00 * move_start_position_s)
        move_end_position_s = move_end_position_s + (0.00* move_end_position_s)
        move_start_vel_s = move_start_position_s + (0.0 * move_start_position_s)
        move_end_vel_s = move_end_position_s - (0.0 * move_end_position_s)
    
        print("\nManual position-window adjustment applied:")
        print(f"  theta/settling move_start_s       : {move_start_theta_s:.6f}")
        print(f"  theta/settling move_end_s         : {move_end_theta_s:.6f}")
        print(f"  position/placement move_start_s   : {move_start_position_s:.6f}")
        print(f"  position/placement move_end_s     : {move_end_position_s:.6f}")
        print(f"  velocity move_start_s   : {move_start_vel_s:.6f}")
        print(f"  velocity move_end_s     : {move_end_vel_s:.6f}")

   


    free_decay_start_s = move_end_theta_s
    free_decay_info = motion_info
    command_end_time_s = move_end_theta_s

    if use_stop_deadband and np.isfinite(move_end_detected_s):
        pivot_vx_in_s_report_final, deadband_info = apply_stop_deadband(
            time_s=time_s, pos_in=pivot_x_in_report_smooth, vel_in_s=pivot_vx_in_s_report_smooth,
            command_end_time_s=move_end_detected_s, buffer_s=stop_deadband_buffer_s,
            late_noise_t0=max(command_end_time_s + 10.0, 20.0), sigma_mult=stop_deadband_sigma_mult,
            final_pos_window=101, final_pos_tol_in=stop_deadband_final_pos_tol_in)
    else:
        pivot_vx_in_s_report_final = pivot_vx_in_s_report_smooth.copy()
        deadband_info = None

    velocity_info = compute_motion_velocity_metrics(
            time_s=time_s,
            x_smooth_in=pivot_x_in_report_smooth,
            vx_smooth_in_s=pivot_vx_in_s_report_smooth,
            move_start_s=move_start_vel_s,
            move_end_s=move_end_vel_s
        )

    # Diagnostic theta using marker centers only, before pivot-offset correction.
    # This helps identify whether a theta spike is from raw tracking or correction logic.
    dx_markercenter = bob_x - pivot_x
    dy_markercenter = bob_y - pivot_y
    
    theta_markercenter_raw_rad = np.full_like(time_s, np.nan, dtype=float)
    theta_markercenter_raw_rad[theta_valid] = np.arctan2(
        dx_markercenter[theta_valid],
        dy_markercenter[theta_valid]
    )
    theta_markercenter_raw_deg = np.degrees(theta_markercenter_raw_rad)
    
    # Corrected theta using corrected physical pivot.
    dx = bob_x - pivot_x_corr
    dy = bob_y - pivot_y_corr
    L_px = np.full_like(time_s, np.nan, dtype=float)
    L_px[theta_valid] = np.sqrt(dx[theta_valid] ** 2 + dy[theta_valid] ** 2)
    L_px_mean = mean_first_n_valid(L_px, theta_valid, 50)
    L_px_norm = np.full_like(L_px, np.nan, dtype=float) if (not np.isfinite(L_px_mean) or L_px_mean == 0) else L_px / L_px_mean

    theta_raw_rad = np.full_like(time_s, np.nan, dtype=float)
    theta_raw_rad[theta_valid] = np.arctan2(dx[theta_valid], dy[theta_valid])
    theta_raw_deg = np.degrees(theta_raw_rad)

    theta_center_deg = moving_average_nan(theta_raw_deg, int(analysis_defaults.get("theta_center_smooth_window", 1001)))
    mask_bias = theta_valid & np.isfinite(theta_center_deg) & np.isfinite(pivot_x_corr) & np.isfinite(time_s)
    if np.sum(mask_bias) >= 20:
        x_b = pivot_x_corr[mask_bias]
        t_b = time_s[mask_bias]
        A = np.column_stack([x_b**2, x_b, t_b, np.ones_like(x_b)])
        coef_xt, _, _, _ = np.linalg.lstsq(A, theta_center_deg[mask_bias], rcond=None)
        theta_bias_x_deg = (coef_xt[0] * pivot_x_corr**2
                            + coef_xt[1] * pivot_x_corr
                            + coef_xt[2] * time_s
                            + coef_xt[3])
        coef = coef_xt  # keep coef defined so the diagnostic block below still works
        print(f"\nEnhanced geometry+time correction")
        print(f"  coef[0] (x^2 term) : {coef[0]:.6e}")
        print(f"  coef[1] (x^1 term) : {coef[1]:.6f}")
        print(f"  coef[2] (t term)   : {coef[2]:.6f}")
        print(f"  coef[3] (constant) : {coef[3]:.6f}")
        if abs(coef_xt[2]) > 0.005:
            print("  NOTE: significant temporal drift in theta center detected and removed")
    else:
        theta_bias_x_deg = np.full_like(theta_raw_deg, np.nan)
        coef = np.array([0.0, 0.0, 0.0])

    theta_geomcorr_deg = theta_raw_deg.copy()
    good_bias = np.isfinite(theta_bias_x_deg)
    theta_geomcorr_deg[good_bias] = theta_raw_deg[good_bias] - theta_bias_x_deg[good_bias]

    theta_zero_method = analysis_defaults.get("theta_zero_method", "first_n_frames")
    theta_zero_n_frames = int(analysis_defaults.get("theta_zero_n_frames", 20))
    theta_manual_offset_deg = float(analysis_defaults.get("theta_manual_offset_deg", 0.0))
    theta_for_zeroing_deg = theta_geomcorr_deg

    if theta_zero_method == "first_n_frames":
        theta_bias_deg = mean_first_n_valid(theta_for_zeroing_deg, theta_valid, theta_zero_n_frames)
    elif theta_zero_method == "late_window":
        late_t0 = float(analysis_defaults.get("theta_bias_t0_s", 10.0))
        late_t1 = float(analysis_defaults.get("theta_bias_t1_s", 60.0))
        theta_bias_deg = mean_in_time_window(theta_for_zeroing_deg, time_s, theta_valid, late_t0, late_t1)
        if not np.isfinite(theta_bias_deg):
            print("WARNING: late_window theta bias failed, falling back to first_n_frames.")
            theta_bias_deg = mean_first_n_valid(theta_for_zeroing_deg, theta_valid, theta_zero_n_frames)
    elif theta_zero_method == "none":
        theta_bias_deg = 0.0
    else:
        print(f"WARNING: Unknown theta_zero_method '{theta_zero_method}', using first_n_frames.")
        theta_bias_deg = mean_first_n_valid(theta_for_zeroing_deg, theta_valid, theta_zero_n_frames)
    if not np.isfinite(theta_bias_deg):
        theta_bias_deg = 0.0

    theta_zeroed_before_noise_deg = theta_for_zeroing_deg - theta_bias_deg - theta_manual_offset_deg
    theta_zeroed_deg = theta_zeroed_before_noise_deg.copy()
    
    if (not is_stationary_noise_run) and stationary_profile is not None:
        theta_zeroed_deg = theta_zeroed_deg - stationary_theta_mean_subtracted_deg
    
    # --------------------------------------------------------
    # Optional scalar post-stop theta centering
    # --------------------------------------------------------
    use_poststop_theta_centering = parse_bool(
        analysis_defaults.get("use_poststop_theta_centering", True),
        default=True
    )
    
    theta_poststop_center_bias_deg = 0.0
    theta_poststop_center_info = {
        "method": "disabled",
        "theta_poststop_center_bias_deg": 0.0,
        "n_samples": 0
    }
    
    if use_poststop_theta_centering and (not is_stationary_noise_run):
        theta_poststop_center_bias_deg, theta_poststop_center_info = estimate_poststop_theta_center_bias(
            time_s=time_s,
            theta_deg=theta_zeroed_deg,
            theta_valid=theta_valid,
            move_end_s=move_end_detected_s,
            delay_s=float(analysis_defaults.get("theta_poststop_center_delay_s", 2.0)),
            min_samples=int(analysis_defaults.get("theta_poststop_center_min_samples", 100)),
            max_abs_deg=float(analysis_defaults.get("theta_poststop_center_max_abs_deg", 5.0))
        )
    
        theta_zeroed_deg = theta_zeroed_deg - theta_poststop_center_bias_deg

    if is_stationary_noise_run:
        stationary_profile_out = build_stationary_noise_profile(
            time_s=time_s, theta_deg=theta_zeroed_deg, vx_in_s=pivot_vx_in_s_report_final,
            x_in=pivot_x_in_report_smooth, theta_valid=theta_valid, pivot_valid=pivot_valid,
            meta={"config_path": os.path.abspath(config_path), "raw_csv_path": os.path.abspath(raw_csv_path),
                  "theta_zero_method": theta_zero_method, "theta_bias_deg": theta_bias_deg,
                  "velocity_smoothing_method_used": velocity_smoothing_method_used})
        print("\nSaved stationary noise profile:")
        print("  theta_mean_deg :", stationary_profile_out.get("theta_mean_deg"))
        print("  theta_std_deg  :", stationary_profile_out.get("theta_std_deg"))
        print("  vx_std_in_s    :", stationary_profile_out.get("vx_std_in_s"))

    
    # ============================================================
    # DIAGNOSTIC BLOCK: Geometry correction coefficients + FFT
    # ============================================================
    
    # 1. Print geometry correction coefficients
    print("\n--- Geometry correction diagnostic ---")
    if np.sum(mask_bias) >= 10:
        print(f"  quadratic fit coef (a, b, c): {coef}")
        print(f"  coef[0] (x^2 term): {coef[0]:.6f}")
        print(f"  coef[1] (x^1 term): {coef[1]:.6f}")
        print(f"  coef[2] (constant): {coef[2]:.6f}")
        print(f"  pivot_x_corr range: {np.nanmin(pivot_x_corr[mask_bias]):.2f} to {np.nanmax(pivot_x_corr[mask_bias]):.2f} px")
        correction_range = np.nanmax(theta_bias_x_deg[mask_bias]) - np.nanmin(theta_bias_x_deg[mask_bias])
        print(f"  implied correction range [deg]: {correction_range:.6f}")
        if correction_range < 0.05:
            print("  WARNING: geometry correction is effectively zero — coef are near-flat.")
    else:
        print(f"  WARNING: mask_bias only has {np.sum(mask_bias)} valid samples — fit was skipped or unreliable.")
    
    
    # 2. FFT of theta_zeroed_deg in the post-stop window
    print("\n--- FFT diagnostic (post-stop theta) ---")
    fft_start_s = move_end_detected_s if np.isfinite(move_end_detected_s) else 0.0
    fft_mask = np.isfinite(time_s) & np.isfinite(theta_zeroed_deg) & (time_s >= fft_start_s)
    
    if np.sum(fft_mask) >= 64:
        t_fft = time_s[fft_mask]
        th_fft = theta_zeroed_deg[fft_mask]
    
        # Interpolate to uniform grid in case of any gaps
        dt_fft = float(np.nanmedian(np.diff(t_fft)))
        t_uniform = np.arange(t_fft[0], t_fft[-1], dt_fft)
        th_uniform = np.interp(t_uniform, t_fft, th_fft)
    
        # Detrend and window to reduce spectral leakage
        th_detrended = th_uniform - np.mean(th_uniform)
        window = np.hanning(len(th_detrended))
        th_windowed = th_detrended * window
    
        N = len(th_windowed)
        fft_vals = np.fft.rfft(th_windowed)
        fft_freqs = np.fft.rfftfreq(N, d=dt_fft)
        fft_mag = np.abs(fft_vals)
    
        # Find top 5 peaks in the FFT
        from scipy.signal import find_peaks as _find_peaks
        fft_peaks, _ = _find_peaks(fft_mag, prominence=np.nanmax(fft_mag) * 0.05)
        top_n = min(5, len(fft_peaks))
        if top_n > 0:
            top_idx = fft_peaks[np.argsort(fft_mag[fft_peaks])[::-1][:top_n]]
            print(f"  Top {top_n} spectral peaks:")
            for rank, idx in enumerate(top_idx):
                print(f"    [{rank+1}] f = {fft_freqs[idx]:.4f} Hz  |  T = {1/fft_freqs[idx]:.3f} s  |  magnitude = {fft_mag[idx]:.4f}")
            dominant_f = fft_freqs[top_idx[0]]
            if top_n >= 2:
                f1 = fft_freqs[top_idx[0]]
                f2 = fft_freqs[top_idx[1]]
                beat_period = 1.0 / abs(f1 - f2) if abs(f1 - f2) > 1e-6 else np.inf
                print(f"  Frequency separation (f1-f2): {abs(f1-f2):.4f} Hz")
                print(f"  Implied beat period: {beat_period:.2f} s")
        else:
            print("  No clear spectral peaks found.")
            dominant_f = np.nan
    
        # Plot FFT
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freqs, fft_mag, label="FFT magnitude")
        if top_n > 0:
            plt.plot(fft_freqs[top_idx], fft_mag[top_idx], "ro", label="top peaks")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.title("FFT of Post-Stop Theta (Hanning windowed)")
        plt.xlim(0, min(5.0, fft_freqs[-1]))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(make_tagged_name_from_trial_stem(config_path, "_FFT"), dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print(f"  Not enough post-stop samples for FFT ({np.sum(fft_mask)} found).")
 


    ideal_x_in, ideal_info = generate_idealized_position_trace(
        time_s,
        pivot_x_in_report_smooth,
        move_start_position_s,
        move_end_position_s,
        command_speed_in_s=command_speed_in_s
    )
    
    # Compare the final measured position plateau against the idealized target.
    # This is the placement/speed-ideal error that matches the position plot.
    delta_x_in, placement_info = compute_placement_metrics(
        time_s,
        pivot_x_in_report_smooth,
        ideal_x_in,
        move_end_position_s,
        goal_in=placement_goal_in,
        poststop_buffer_s=placement_poststop_buffer_s
    )
    
 

    if no_motion_detected:
        decay_metrics = empty_decay_metrics(free_decay_start_s)
        t_settle_band_s_measured = np.nan
        settled_observed = False
        settle_censor_lb_s = np.nan
        t_settle_after_motion_end_s = np.nan
    else:
        decay_metrics = extract_free_decay_metrics(
            time_s=time_s, theta_deg=theta_zeroed_deg, free_decay_start_s=free_decay_start_s,
            command_end_time_s=free_decay_start_s, theta_band_deg=theta_band_deg,
            min_peak_deg=float(analysis_defaults.get("metrics_min_peak_deg", 0.5)),
            min_peak_prom_deg=float(analysis_defaults.get("metrics_min_peak_prom_deg", 0.15)),
            min_peak_spacing_s=float(analysis_defaults.get("metrics_min_peak_spacing_s", 0.15)))
        t_settle_band_s_measured, settled_observed, settle_censor_lb_s = measured_deadband_settling_with_dwell(
            time_s=time_s, theta_deg=theta_zeroed_deg, start_time_s=move_end_theta_s,
            band_deg=theta_band_deg, dwell_s=settling_deadband_dwell_s)
        
        t_settle_after_motion_end_s = (
            t_settle_band_s_measured - move_end_theta_s
            if settled_observed and np.isfinite(t_settle_band_s_measured) and np.isfinite(move_end_theta_s)
            else np.nan
        )
        
        t_settle_envelope_s, envelope_settled_observed, envelope_censor_lb_s, envelope_info = envelope_settling_time_from_peaks(
        time_s=time_s,
        theta_deg=theta_zeroed_deg,
        start_time_s=move_end_theta_s,
        band_deg=theta_band_deg,
        min_peak_prom_deg=float(analysis_defaults.get("envelope_min_peak_prom_deg", 0.05)),
        min_peak_spacing_s=float(analysis_defaults.get("envelope_min_peak_spacing_s", 0.5))
    )
    
    t_settle_envelope_after_motion_end_s = (
        t_settle_envelope_s - move_end_theta_s
        if envelope_settled_observed and np.isfinite(t_settle_envelope_s) and np.isfinite(move_end_theta_s)
        else np.nan
    )

    # Strict settling: first time after motion end where all remaining samples
    # stay inside the deadband. This is more conservative than dwell settling.
    t_settle_band_strict_s = find_band_settling_time(
        time_s=time_s,
        theta_deg=theta_zeroed_deg,
        start_time_s=move_end_theta_s,
        band_deg=theta_band_deg
    )
    
    t_settle_strict_after_motion_end_s = (
        t_settle_band_strict_s - move_end_detected_s
        if np.isfinite(t_settle_band_strict_s) and np.isfinite(move_end_detected_s)
        else np.nan
    )
    # Final reported travel should come from the final position plateau,
    # not from the position at motion_end_detected_s.
    final_position_window_n_frames = int(analysis_defaults.get("final_position_window_n_frames", 100))
    
    travel_measured_in = mean_last_n_valid(
        pivot_x_in_report_smooth,
        np.isfinite(pivot_x_in_report_smooth),
        final_position_window_n_frames
    )
    
    idealized_target_in = ideal_info.get("ideal_target_in", np.nan)
    
    if np.isfinite(travel_measured_in) and np.isfinite(idealized_target_in):
        final_minus_ideal_target_in = travel_measured_in - idealized_target_in
        final_minus_ideal_target_abs_in = abs(final_minus_ideal_target_in)
    else:
        final_minus_ideal_target_in = np.nan
        final_minus_ideal_target_abs_in = np.nan
    
    max_abs_theta_deg = np.nanmax(np.abs(theta_zeroed_deg)) if np.any(np.isfinite(theta_zeroed_deg)) else np.nan
    max_speed_in_s = np.nanmax(np.abs(pivot_vx_in_s_report_final)) if np.any(np.isfinite(pivot_vx_in_s_report_final)) else np.nan

    max_abs_theta_markercenter_deg = (
    np.nanmax(np.abs(theta_markercenter_raw_deg))
    if np.any(np.isfinite(theta_markercenter_raw_deg))
    else np.nan
)
    
    if not no_motion_detected and np.isfinite(max_abs_theta_deg):
        t_settle_2pct_of_max_s, theta_2pct_of_max_deg, settle_2pct_of_max_observed, settle_2pct_of_max_censor_lb_s = settling_time_from_fraction_of_ref(
            time_s=time_s, theta_deg=theta_zeroed_deg, start_time_s=move_end_theta_s,
            ref_amp_deg=max_abs_theta_deg, frac=0.02, require_sustained=True)
    else:
        t_settle_2pct_of_max_s, theta_2pct_of_max_deg, settle_2pct_of_max_observed, settle_2pct_of_max_censor_lb_s = (np.nan, np.nan, False, np.nan)

    print("\nAnalysis settings:")
    print(f"  run_type                         : {'stationary_noise' if is_stationary_noise_run else 'regular_trial'}")
    print(f"  direction_sign                   : {direction_sign:+.0f}")
    print(f"  velocity_smoothing_method_used   : {velocity_smoothing_method_used}")
    print(f"  theta_zero_method                : {theta_zero_method}")
    print(f"  theta bias used [deg]            : {theta_bias_deg:.6f}")
    print(f"  stationary theta mean subtract   : {stationary_theta_mean_subtracted_deg:+.6f}")
    print(f"  poststop theta center method     : {theta_poststop_center_info.get('method', '')}")
    print(f"  poststop theta center bias [deg] : {theta_poststop_center_bias_deg:+.6f}")
    print(f"  poststop theta center samples    : {theta_poststop_center_info.get('n_samples', 0)}")   
    print(f"  theta deadband [deg]             : {theta_band_deg:.6f}")
    print(f"  theta deadband dwell [s]         : {settling_deadband_dwell_s:.6f}")
    print(f"  command_speed_in_s               : {command_speed_in_s:.6f}")
    print(f"  placement_goal_in                : {placement_goal_in:.6f}")
    if np.isfinite(travel_measured_in):
        print(f"  reported net travel [in]         : {travel_measured_in:.6f}")

    print("\nMotion / placement summary:")
    print(f"  motion_info method               : {motion_info.get('method', '')}")
    print(f"  no_motion_detected               : {int(no_motion_detected)}")
    print(f"  move_start_detected_s            : {move_start_detected_s if np.isfinite(move_start_detected_s) else 'N/A'}")
    print(f"  move_end_detected_s              : {move_end_detected_s if np.isfinite(move_end_detected_s) else 'N/A'}")
    print(f"  move duration [s]                : {velocity_info.get('move_duration_s')}")
    print(f"  travel at detected motion end [in]: {velocity_info.get('move_travel_in')}")
    print(f"  reported final net travel [in]   : {travel_measured_in:.6f}")
    print(f"  idealized target [in]            : {idealized_target_in}")
    print(f"  final - ideal target [in]        : {final_minus_ideal_target_in}")
    print(f"  |final - ideal target| [in]      : {final_minus_ideal_target_abs_in}")
    print(f"  placement mean error [in]        : {placement_info.get('placement_error_mean_poststop_in')}")
    print(f"  placement maxabs error [in]      : {placement_info.get('placement_error_maxabs_poststop_in')}")
    print(f"  placement pass <= +/-1/8 in      : {placement_info.get('placement_pass')}")
    print(f"  mean velocity travel/duration    : {velocity_info.get('mean_velocity_from_travel_in_s')}")
    print(f"  mean velocity signal [in/s]      : {velocity_info.get('mean_velocity_signal_in_s')}")
    print(f"  median velocity signal [in/s]    : {velocity_info.get('median_velocity_signal_in_s')}")

    print("\nQuick summary:")
    print(f"  Max |theta| [deg]                : {max_abs_theta_deg:.6f}")
    print(f"  Max |pivot vx| [in/s]            : {max_speed_in_s:.6f}")
    
    print(f"  Max |marker-center theta| [deg]  : {max_abs_theta_markercenter_deg:.6f}")
    

    
    if settled_observed:
        print(f"  Deadband settling after end [s]  : {t_settle_after_motion_end_s:.6f}")
    elif np.isfinite(settle_censor_lb_s):
        print(f"  Deadband settling after end [s]  : >{settle_censor_lb_s:.6f} (not observed)")
    else:
        print("  Deadband settling after end [s]  : N/A")
        
    if np.isfinite(t_settle_strict_after_motion_end_s):
        print(f"  Strict final settling after end [s]: {t_settle_strict_after_motion_end_s:.6f}")
    else:
        print("  Strict final settling after end [s]: not observed")
    
    if envelope_settled_observed:
        print(f"  Envelope settling after end [s]  : {t_settle_envelope_after_motion_end_s:.6f}")
    elif np.isfinite(envelope_censor_lb_s):
        print(f"  Envelope settling after end [s]  : >{envelope_censor_lb_s:.6f} (not observed)")
    else:
        print("  Envelope settling after end [s]  : N/A")
    
    print(f"  First envelope peak [deg]        : {envelope_info.get('first_peak_deg')}")
    print(f"  Last envelope peak [deg]         : {envelope_info.get('last_peak_deg')}")
    print(f"  Number of envelope peaks         : {envelope_info.get('n_envelope_peaks')}")

    header = ",".join([
        "frame_idx", "time_s", "pivot_x_roi", "pivot_y_roi", "bob_x_roi", "bob_y_roi", "pivot_found", "bob_found",
        "pivot_x_corr_roi", "pivot_y_corr_roi", "pivot_x_in_raw", "pivot_y_in_raw", "pivot_x_in_rel",
        "pivot_x_in_report", "pivot_x_in_report_smooth", "ideal_x_in", "delta_x_in", "bob_x_in_raw", "bob_y_in_raw",
        "theta_markercenter_raw_deg", "theta_raw_deg", "theta_geomcorr_deg", "theta_zeroed_before_noise_deg", "theta_zeroed_deg",
        "pivot_vx_in_s_raw", "pivot_vx_in_s_report", "pivot_vx_in_s_report_smooth", "pivot_vx_in_s_report_final",
        "direction_sign", "pivot_dx_px_used", "pivot_dy_px_used"])

    out_matrix = np.column_stack([
        frame_idx, time_s, pivot_x, pivot_y, bob_x, bob_y, pivot_found, bob_found, pivot_x_corr, pivot_y_corr,
        pivot_x_in_raw, pivot_y_in_raw, pivot_x_in_rel, pivot_x_in_report, pivot_x_in_report_smooth,
        ideal_x_in, delta_x_in, bob_x_in_raw, bob_y_in_raw, theta_markercenter_raw_deg, theta_raw_deg, theta_geomcorr_deg,
        theta_zeroed_before_noise_deg, theta_zeroed_deg, pivot_vx_in_s_raw, pivot_vx_in_s_report,
        pivot_vx_in_s_report_smooth, pivot_vx_in_s_report_final, np.full_like(time_s, direction_sign, dtype=float),
        pivot_dx_px, pivot_dy_px])
    np.savetxt(out_csv, out_matrix, delimiter=",", header=header, comments="", fmt="%.10f")

    metrics_csv_path = os.path.join(os.path.dirname(os.path.abspath(out_csv)), "trial_metrics_master.csv")
    condition_id, trial_num, config_stem = parse_condition_and_trial(config_path)
    trial_row = {
        "config_stem": config_stem, "condition_id": condition_id, "trial_num": trial_num,
        "run_type": "stationary_noise" if is_stationary_noise_run else "regular_trial",
        "video_path": cfg.get("video_path", ""), "raw_csv": os.path.abspath(raw_csv_path), "out_csv": os.path.abspath(out_csv),
        "pivot_px_source": pivot_px_source, "bob_px_source": bob_px_source, "pivot_offset_mode_used": pivot_offset_mode_used,
        "velocity_smoothing_method_used": velocity_smoothing_method_used,
        "stationary_noise_profile_path_loaded": stationary_profile_path_used,
        "stationary_theta_mean_subtracted_deg": stationary_theta_mean_subtracted_deg,
        "no_motion_detected": int(no_motion_detected), "no_motion_travel_threshold_in": no_motion_travel_threshold_in,
        "motion_window_method": motion_info.get("method", ""),
        "move_start_detected_s": move_start_detected_s, "move_end_detected_s": move_end_detected_s,
        "command_end_time_s": command_end_time_s, "free_decay_start_s": decay_metrics["free_decay_start_s"],
        "free_decay_start_method": free_decay_info.get("method", ""),
        "free_decay_vel_thresh_in_s": free_decay_info.get("vel_thresh_in_s", np.nan),
        "command_speed_in_s": command_speed_in_s, "ideal_position_method": ideal_info.get("method", ""),
        "ideal_x0_in": ideal_info.get("ideal_x0_in", np.nan), "ideal_target_in": ideal_info.get("ideal_target_in", np.nan),
        "ideal_travel_in": ideal_info.get("ideal_travel_in", np.nan),
        "placement_error_mean_poststop_in": placement_info.get("placement_error_mean_poststop_in", np.nan),
        "placement_error_maxabs_poststop_in": placement_info.get("placement_error_maxabs_poststop_in", np.nan),
        "placement_goal_in": placement_info.get("placement_goal_in", placement_goal_in),
        "placement_pass": placement_info.get("placement_pass", np.nan),
        "placement_n_poststop_samples": placement_info.get("placement_n_poststop_samples", 0),
        "reported_net_travel_in": travel_measured_in,
        "max_abs_theta_deg_full_record": max_abs_theta_deg, "max_abs_speed_in_s_full_record": max_speed_in_s,
        "first_peak_abs_deg": decay_metrics["first_peak_abs_deg"], "max_poststop_abs_deg": decay_metrics["max_poststop_abs_deg"],
        "n_abs_peaks": decay_metrics["n_abs_peaks"], "n_same_phase_peaks": decay_metrics["n_same_phase_peaks"],
        "Td_s": decay_metrics["Td_s"], "fd_hz": decay_metrics["fd_hz"], "wd_rad_s": decay_metrics["wd_rad_s"],
        "zeta_logdec": decay_metrics["zeta_logdec"], "wn_rad_s": decay_metrics["wn_rad_s"],
        "t_settle_band_deg": float(theta_band_deg), "t_settle_deadband_dwell_s": settling_deadband_dwell_s,
        "t_settle_band_s_measured": t_settle_band_s_measured,
        "t_settle_band_after_motion_end_s": t_settle_after_motion_end_s,
        "t_settle_band_observed": int(settled_observed),
        "t_settle_band_censored_lower_bound_s": settle_censor_lb_s,
        "t_settle_band_report": (f"{t_settle_after_motion_end_s:.6f}" if settled_observed else (f">{settle_censor_lb_s:.6f}" if np.isfinite(settle_censor_lb_s) else "N/A")),
        "t_settle_2pct_s_est": decay_metrics["t_settle_2pct_s_est"],
        "theta_2pct_of_max_deg": theta_2pct_of_max_deg, "t_settle_2pct_of_max_s": t_settle_2pct_of_max_s,
        "t_settle_2pct_of_max_observed": int(settle_2pct_of_max_observed),
        "t_settle_2pct_of_max_censored_lower_bound_s": settle_2pct_of_max_censor_lb_s,
        "t_settle_2pct_of_max_report": (f"{t_settle_2pct_of_max_s:.6f}" if settle_2pct_of_max_observed else (f">{settle_2pct_of_max_censor_lb_s:.6f}" if np.isfinite(settle_2pct_of_max_censor_lb_s) else "N/A")),
        "move_duration_s": velocity_info.get("move_duration_s", np.nan),
        "move_x_start_in": velocity_info.get("move_x_start_in", np.nan),
        "move_x_end_in": velocity_info.get("move_x_end_in", np.nan),
        "move_travel_in": velocity_info.get("move_travel_in", np.nan),
        "mean_velocity_from_travel_in_s": velocity_info.get("mean_velocity_from_travel_in_s", np.nan),
        "mean_velocity_signal_in_s": velocity_info.get("mean_velocity_signal_in_s", np.nan),
        "median_velocity_signal_in_s": velocity_info.get("median_velocity_signal_in_s", np.nan),
        "mean_abs_velocity_signal_in_s": velocity_info.get("mean_abs_velocity_signal_in_s", np.nan),
        "velocity_signal_std_in_s": velocity_info.get("velocity_signal_std_in_s", np.nan),
        "velocity_n_samples": velocity_info.get("velocity_n_samples", 0),
        "t_settle_band_strict_s": t_settle_band_strict_s,
        "t_settle_strict_after_motion_end_s": t_settle_strict_after_motion_end_s,
        "t_settle_strict_observed": int(np.isfinite(t_settle_band_strict_s)),
        "t_settle_envelope_s": t_settle_envelope_s,
        "t_settle_envelope_after_motion_end_s": t_settle_envelope_after_motion_end_s,
        "t_settle_envelope_observed": int(envelope_settled_observed),
        "t_settle_envelope_censored_lower_bound_s": envelope_censor_lb_s,
        "first_envelope_peak_deg": envelope_info.get("first_peak_deg", np.nan),
        "last_envelope_peak_deg": envelope_info.get("last_peak_deg", np.nan),
        "max_envelope_peak_deg": envelope_info.get("max_envelope_peak_deg", np.nan),
        "n_envelope_peaks": envelope_info.get("n_envelope_peaks", 0),
        
        "reported_final_net_travel_in": travel_measured_in,
        "idealized_target_in": idealized_target_in,
        "final_minus_ideal_target_in": final_minus_ideal_target_in,
        "final_minus_ideal_target_abs_in": final_minus_ideal_target_abs_in,
        "travel_at_detected_motion_end_in": velocity_info.get("move_travel_in", np.nan),    
   }
    if is_stationary_noise_run:
        trial_row.update({
            "stationary_noise_profile_path_saved": os.path.abspath(noise_profile_out_path),
            "stationary_theta_mean_deg": stationary_profile_out.get("theta_mean_deg", np.nan),
            "stationary_theta_std_deg": stationary_profile_out.get("theta_std_deg", np.nan),
            "stationary_theta_maxabs_deg": stationary_profile_out.get("theta_maxabs_deg", np.nan),
            "stationary_theta_ptp_deg": stationary_profile_out.get("theta_ptp_deg", np.nan),
            "stationary_vx_std_in_s": stationary_profile_out.get("vx_std_in_s", np.nan),
            "stationary_vx_maxabs_in_s": stationary_profile_out.get("vx_maxabs_in_s", np.nan),
            "stationary_x_std_in": stationary_profile_out.get("x_std_in", np.nan),
        })
    append_metrics_row(metrics_csv_path, trial_row)

    print("\nSaved analyzed CSV:")
    print(f"  {os.path.abspath(out_csv)}")
    print("\nAppended trial metrics CSV:")
    print(f"  {os.path.abspath(metrics_csv_path)}")

    plt.figure(figsize=(12, 6))
    plt.plot(time_s, theta_zeroed_deg, label="theta")
    plt.axhline(theta_band_deg, color="k", linestyle="--", linewidth=1, label=f"+/-{theta_band_deg:g} deg deadband + 3σ")
    plt.axhline(-theta_band_deg, color="k", linestyle="--", linewidth=1)
    plt.axhline(2, color="r", linestyle="--", linewidth=1, label="+/-2 deg constraint")
    #plt.axhline(0, color="r", linestyle="--", linewidth=2, label="ZERO")
    plt.axhline(-2, color="r", linestyle="--", linewidth=1)
    plot_vline_if_finite(move_start_theta_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
    plot_vline_if_finite(move_end_theta_s, color="k", linestyle="-.", linewidth=2, label="Motion End")
    plt.xlabel("Time [s]"); plt.ylabel("Sway angle [deg]"); plt.title("Pendulum Sway Angle vs Time")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(angle_plot_path, dpi=300)
    if show_plots: plt.show()
    else: plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, pivot_x_in_report, label="tracked raw")
    plt.plot(time_s, pivot_x_in_report_smooth, label="tracked smoothed")
    if np.any(np.isfinite(ideal_x_in)):
        plt.plot(time_s, ideal_x_in, linestyle="--", label="idealized 4.6 in/s")
    plot_vline_if_finite(move_start_detected_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
    plot_vline_if_finite(move_end_detected_s, color="k", linestyle="-.", linewidth=2, label="Motion End")
    plt.xlabel("Time [s]"); plt.ylabel("Pivot x-position [in]"); plt.title("Pivot Horizontal Position vs Time")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(pos_plot_path, dpi=300)
    if show_plots: plt.show()
    else: plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_s, pivot_vx_in_s_report_final, label="final speed")
    if np.isfinite(command_speed_in_s):
        plt.axhline(command_speed_in_s, linestyle=":", linewidth=2, label=f"command {command_speed_in_s:g} in/s")
    
    mean_v_travel = velocity_info.get("mean_velocity_from_travel_in_s", np.nan)
    if np.isfinite(mean_v_travel):
        plt.axhline(mean_v_travel, linestyle="--", linewidth=1, label=f"mean {mean_v_travel:.2f} in/s")
    plot_vline_if_finite(move_start_detected_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
    plot_vline_if_finite(move_end_detected_s, color="k", linestyle="-.", linewidth=2, label="Motion End")
    plt.xlabel("Time [s]"); plt.ylabel("Pivot horizontal speed [in/s]"); plt.title("Pivot Horizontal Speed vs Time")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(speed_plot_path, dpi=300)
    if show_plots: plt.show()
    else: plt.close()
    
    # Diagnostic: Theta
    plt.figure(figsize=(10, 5))
    plt.plot(time_s, theta_raw_deg, label="theta raw deg", alpha=0.75)
    plt.plot(time_s, theta_geomcorr_deg, label="theta geom-corrected", alpha=0.85)
    plt.plot(time_s, theta_zeroed_deg, label="theta final", linewidth=2)
    
    plt.axhline(theta_band_deg, color="k", linestyle="--", linewidth=1, label=f"+/-{theta_band_deg:g} deg deadband + 3σ")
    plt.axhline(-theta_band_deg, color="k", linestyle="--", linewidth=1)
    plt.axhline(2.0, color="r", linestyle="--", linewidth=1, label="+/-2 deg constraint")
    plt.axhline(-2.0, color="r", linestyle="--", linewidth=1)
    
    plot_vline_if_finite(move_start_theta_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
    plot_vline_if_finite(move_end_theta_s, color="k", linestyle="-.", linewidth=2, label="Motion End")
    
    plt.xlabel("Time [s]")
    plt.ylabel("Sway angle [deg]")
    plt.title("Theta Comparison: Raw vs Corrected vs Final")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(make_tagged_name_from_trial_stem(config_path, "_THETA_COMPARE"), dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
        
 
    bob_rel_x_in = np.full_like(time_s, np.nan, dtype=float)
    bob_rel_x_in[theta_valid] = (bob_x[theta_valid] - pivot_x_corr[theta_valid]) / px_per_in_bob
    
    # Estimate the static/equilibrium center after the move.
    bob_center_delay_s = 2.0
    bob_center_mask = (
        theta_valid
        & np.isfinite(bob_rel_x_in)
        & np.isfinite(time_s)
        & (time_s >= move_end_detected_s + bob_center_delay_s)
    )
    
    if np.sum(bob_center_mask) >= 100:
        bob_rel_x_center_in = float(np.nanmedian(bob_rel_x_in[bob_center_mask]))
    else:
        bob_rel_x_center_in = float(np.nanmedian(bob_rel_x_in[np.isfinite(bob_rel_x_in)]))
    
    bob_rel_x_centered_in = bob_rel_x_in - bob_rel_x_center_in
    
    # Estimate lateral displacement equivalent to the angle bands
    L_in_est = np.nan
    if np.isfinite(L_px_mean) and px_per_in_bob > 0:
        L_in_est = L_px_mean / px_per_in_bob
    
    deadband_x_in = np.nan
    constraint_x_in = np.nan
    if np.isfinite(L_in_est):
        deadband_x_in = L_in_est * np.tan(np.deg2rad(theta_band_deg))
        constraint_x_in = L_in_est * np.tan(np.deg2rad(2.0))
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_s, bob_rel_x_centered_in, label="centered bob lateral displacement [in]")
    
    if np.isfinite(deadband_x_in):
        plt.axhline(deadband_x_in, color="k", linestyle="--", linewidth=1, label=f"+/-{theta_band_deg:g} deg equivalent")
        plt.axhline(-deadband_x_in, color="k", linestyle="--", linewidth=1)
    
    if np.isfinite(constraint_x_in):
        plt.axhline(constraint_x_in, color="r", linestyle="--", linewidth=1, label="+/-2 deg equivalent")
        plt.axhline(-constraint_x_in, color="r", linestyle="--", linewidth=1)
    
    plot_vline_if_finite(move_start_theta_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
    plot_vline_if_finite(move_end_theta_s, color="k", linestyle="-.", linewidth=2, label="Motion End")

    plt.xlabel("Time [s]")
    plt.ylabel("Centered relative bob x [in]")
    plt.title("Centered Bob Lateral Displacement Relative to Pivot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(make_tagged_name_from_trial_stem(config_path, "_BOB_RELX_CENTERED"), dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    # Diagnostic: amplitude envelope of final theta
    theta_abs = np.abs(theta_zeroed_deg)
    
    post_mask = (
        np.isfinite(time_s)
        & np.isfinite(theta_abs)
        & np.isfinite(move_end_detected_s)
        & (time_s >= move_end_detected_s)
    )
    
    t_post = time_s[post_mask]
    theta_abs_post = theta_abs[post_mask]
    
    if t_post.size >= 10:
        dt_post = float(np.nanmedian(np.diff(t_post)))
        peak_distance_s = 0.5
        peak_distance_n = max(1, int(round(peak_distance_s / dt_post))) if np.isfinite(dt_post) and dt_post > 0 else 1
    
        env_peaks, _ = find_peaks(
            theta_abs_post,
            prominence=0.05,
            distance=peak_distance_n
        )
    
        plt.figure(figsize=(10, 5))
        plt.plot(t_post, theta_abs_post, label="|theta final|")
        if env_peaks.size > 0:
            plt.plot(t_post[env_peaks], theta_abs_post[env_peaks], "o", label="envelope peaks")
    
        plt.axhline(theta_band_deg, color="k", linestyle="--", linewidth=1, label=f"{theta_band_deg:g} deadband + 3σ")
        plt.xlabel("Time [s]")
        plt.ylabel("|Sway angle| [deg]")
        plt.title("Post-Stop Sway Amplitude Envelope")
        plot_vline_if_finite(move_start_theta_s, color="k", linestyle="--", linewidth=2, label="Motion Start")
        plot_vline_if_finite(move_end_theta_s, color="k", linestyle="-.", linewidth=2, label="Motion End")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(make_tagged_name_from_trial_stem(config_path, "_THETA_ENVELOPE"), dpi=300)
    
        if show_plots:
            plt.show()
        else:
            plt.close()
    
        if env_peaks.size > 0:
            print("\nTheta envelope summary:")
            print(f"  first post-stop envelope peak [deg] : {theta_abs_post[env_peaks[0]]:.6f}")
            print(f"  last post-stop envelope peak [deg]  : {theta_abs_post[env_peaks[-1]]:.6f}")
            print(f"  number of envelope peaks            : {env_peaks.size}")
            print(f"  detected move_start_s            : {move_start_detected_s if np.isfinite(move_start_detected_s) else 'N/A'}")
            print(f"  detected move_end_s              : {move_end_detected_s if np.isfinite(move_end_detected_s) else 'N/A'}")
            print(f"  theta/settling move_start_s      : {move_start_theta_s if np.isfinite(move_start_theta_s) else 'N/A'}")
            print(f"  theta/settling move_end_s        : {move_end_theta_s if np.isfinite(move_end_theta_s) else 'N/A'}")
            print(f"  position/placement move_start_s  : {move_start_position_s if np.isfinite(move_start_position_s) else 'N/A'}")
            print(f"  position/placement move_end_s    : {move_end_position_s if np.isfinite(move_end_position_s) else 'N/A'}")
                    
    print("\nSaved plots:")
    print(f"  {os.path.abspath(angle_plot_path)}")
    print(f"  {os.path.abspath(pos_plot_path)}")
    print(f"  {os.path.abspath(speed_plot_path)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
