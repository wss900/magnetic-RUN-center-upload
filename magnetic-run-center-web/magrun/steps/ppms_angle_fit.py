from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..models import StepMeta, StepOutputs, StepParam
from ..utils.text_parse import pick_col_index, split_columns, _norm


def _fit_func_deg(x_deg: np.ndarray, A: float, B: float, C: float, D: float, E: float, F: float, G: float) -> np.ndarray:
    theta = (x_deg + A) * np.pi / 180.0
    return (
        B * np.cos(theta)
        + C * np.cos(2 * theta) * np.cos(theta)
        + D * np.cos(2 * theta)
        + E * np.sin(theta)
        + F * np.sin(2 * theta)
        + G
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _robust_sigma_mad(x: np.ndarray) -> float:
    """
    Robust scale estimate using MAD (median absolute deviation).
    Returns an estimate of sigma assuming normality, or NaN if undefined.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if (not np.isfinite(mad)) or mad == 0.0:
        return float("nan")
    return 1.4826 * mad


def _spike_filter_by_residual_mad(
    x_deg: np.ndarray,
    y: np.ndarray,
    popt_arr: np.ndarray,
    *,
    k: float = 6.0,
) -> tuple[np.ndarray, int]:
    """
    Robustly drop "spike" points based on residual magnitude.

    Uses MAD of residuals with a configurable cutoff k*sigma.
    """
    y_pred = _fit_func_deg(x_deg, *popt_arr)
    resid = y - y_pred

    # Median-centered residuals for robustness against skew.
    resid_med = float(np.median(resid))
    resid_centered = resid - resid_med
    mad = float(np.median(np.abs(resid_centered)))

    if (not np.isfinite(mad)) or mad == 0.0:
        keep = np.ones_like(resid_centered, dtype=bool)
        removed = int((~keep).sum())
        return keep, removed

    sigma = 1.4826 * mad  # MAD -> std scale (normal assumption)
    keep = np.abs(resid_centered) <= (k * sigma)
    removed = int((~keep).sum())
    return keep, removed


def _spike_filter_by_diff2_mad(
    y: np.ndarray,
    *,
    k: float = 6.0,
) -> tuple[np.ndarray, int]:
    """
    Drop local spike points using second-difference (curvature) outlier detection.

    For monotonic angular scans, lock-in noise spikes often show up as a single
    point with abnormally large second difference:
        d2[i] = y[i+1] - 2*y[i] + y[i-1]
    We flag i where |d2[i]| > k * sigma_d2 (sigma from MAD) and drop point i.
    """
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep, 0

    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    sigma = _robust_sigma_mad(d2)
    if not np.isfinite(sigma):
        return keep, 0

    bad_mid = np.abs(d2 - float(np.median(d2))) > (k * sigma)
    # bad_mid corresponds to indices 1..n-2 in the original y
    if np.any(bad_mid):
        mid_idx = np.nonzero(bad_mid)[0] + 1
        keep[mid_idx] = False
    removed = int((~keep).sum())
    return keep, removed


def _pick_last_sweep_by_reset(
    ang: np.ndarray,
    mag: np.ndarray,
    sig: np.ndarray,
    *,
    zero_window_deg: float = 5.0,
    leave_zero_deg: float = 15.0,
    min_points_between_resets: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick the last angular sweep within a segment.

    A "reset" is detected when angle returns to near 0 (within zero_window_deg)
    after having left the zero region (abs(angle) >= leave_zero_deg). This works
    for both 0→360 and 0→-360 style scans.
    """
    n = int(ang.size)
    if n == 0:
        return ang, mag, sig

    # Track sweep starts by reset events.
    starts: list[int] = [0]
    in_zero = bool(abs(float(ang[0])) <= zero_window_deg)
    has_left_zero = bool(abs(float(ang[0])) >= leave_zero_deg)
    last_reset_i = 0

    for i in range(1, n):
        a = float(ang[i])
        a_abs = abs(a)
        now_in_zero = a_abs <= zero_window_deg
        if a_abs >= leave_zero_deg:
            has_left_zero = True

        # Transition non-zero -> zero after leaving zero: treat as reset
        if (not in_zero) and now_in_zero and has_left_zero and (i - last_reset_i) >= min_points_between_resets:
            starts.append(i)
            last_reset_i = i
            has_left_zero = False

        in_zero = now_in_zero

    if len(starts) <= 1:
        return ang, mag, sig

    s = starts[-1]
    return ang[s:], mag[s:], sig[s:]


def _dedupe_angle_keep_last(
    ang: np.ndarray, mag: np.ndarray, sig: np.ndarray, *, angle_tol_deg: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dedupe repeated angles by binning with a tolerance, keeping the last occurrence.
    """
    if ang.size == 0:
        return ang, mag, sig

    last: dict[float, tuple[int, float, float, float]] = {}
    for i in range(int(ang.size)):
        a = float(ang[i])
        key = round(a / angle_tol_deg) * angle_tol_deg
        last[key] = (i, a, float(mag[i]), float(sig[i]))

    items = sorted(last.values(), key=lambda t: t[0])
    ang2 = np.asarray([t[1] for t in items], dtype=float)
    mag2 = np.asarray([t[2] for t in items], dtype=float)
    sig2 = np.asarray([t[3] for t in items], dtype=float)
    return ang2, mag2, sig2


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
    buf.seek(0)
    return buf.getvalue()


def _parse_float(s: str) -> float | None:
    s2 = s.strip()
    if not s2 or s2.lower() == "nan":
        return None
    try:
        return float(s2)
    except Exception:
        return None


def _parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []

    # find header row: requires mag + angle + some signal
    required = [
        ["mag field", "magnet(oe)", "magnet", "field"],
        ["angle", "position (deg)1", "position (deg)", "position"],
        ["lock-in", "x(v)_sr-830", "y(v)_sr-830", "x(v)_sr-830-1", "y(v)_sr-830-1"],
    ]

    header_idx = -1
    for i, ln in enumerate(lines):
        cols = split_columns(ln)
        cols_norm = [_norm(c) for c in cols]
        ok = True
        for group in required:
            gnorm = [_norm(x) for x in group]
            if not any(any(g in c for c in cols_norm) for g in gnorm):
                ok = False
                break
        if ok:
            header_idx = i
            break

    if header_idx == -1:
        header_idx = 0

    headers = split_columns(lines[header_idx])
    rows = [split_columns(ln) for ln in lines[header_idx + 1 :] if ln.strip()]
    return headers, rows


def _pick_lockin_col(headers: list[str], lockin_channel: str) -> int | None:
    if lockin_channel == "AUTO_LOCKIN_X":
        hnorm = [_norm(h) for h in headers]
        for i, h in enumerate(hnorm):
            if "lock-in" in h and ("x" in h or " x" in h):
                return i
        return None
    return pick_col_index(headers, [lockin_channel])


@dataclass(frozen=True)
class _Cfg:
    lockin_channel: str
    lockin_unit: str
    mag_jump_thresh_oe: float
    min_angle_coverage_deg: float
    min_points_per_segment: int
    sort_angle: bool
    generate_plot: bool
    remove_spikes: bool
    spike_method: str
    spike_k: float
    min_points_after_spike_filter: int


class PpmsAngleFitStep:
    meta = StepMeta(
        id="ppms_angle_fit",
        name="PPMS 角度扫描拟合（多段磁场）",
        category="🧪 PPMS 数据处理",
        description=(
            "兼容两类表头（Mag Field/Angle/Lock-in 或 Magnet(Oe)/Position(Deg)1/X(V)_SR-830）。\n\n"
            "- 按磁场跳变切段（同一文件可含多段不同磁场的角度扫）\n"
            "- 角度模式：支持 0→360 或 0→-360；若同一磁场下多次扫角（回到 0 附近重置），只保留最后一圈\n"
            "- 覆盖不足/点数不足：废弃该段（RejectReason）\n"
            "- 对有效段做七参数拟合并输出 A..G, R2\n"
        ),
        file_types=["txt"],
        params=[
            StepParam(
                key="lockin_channel",
                label="Lock-in 通道（信号列）",
                kind="select",
                default="X(V)_SR-830",
                options=["X(V)_SR-830", "X(V)_SR-830-1", "AUTO_LOCKIN_X"],
            ),
            StepParam(
                key="lockin_unit",
                label="Lock-in 单位（拟合时使用）",
                kind="select",
                default="uV",
                options=["V", "uV"],
                help="选择 uV 会在拟合前将 Volt 乘以 1e6。",
            ),
            StepParam(key="mag_jump_thresh_oe", label="磁场跳变阈值（Oe）", kind="float", default=200.0),
            StepParam(key="min_angle_coverage_deg", label="最小角度覆盖（deg）", kind="float", default=330.0),
            StepParam(key="min_points_per_segment", label="每段最少点数", kind="int", default=30),
            StepParam(key="sort_angle", label="按角度排序", kind="bool", default=True),
            StepParam(key="generate_plot", label="生成合并图（PNG）", kind="bool", default=True),
            StepParam(
                key="remove_spikes",
                label="跳点/尖峰剔除（残差MAD）",
                kind="bool",
                default=True,
                help="将尝试剔除锁相噪声引入的跳点/尖峰（可在下方选择方法）。",
            ),
            StepParam(
                key="spike_method",
                label="跳点剔除方法",
                kind="select",
                default="二阶差分 + 残差MAD（推荐）",
                options=[
                    "二阶差分 + 残差MAD（推荐）",
                    "仅二阶差分（适合单点尖峰）",
                    "仅残差MAD（适合全局离群点）",
                ],
                help="单调扫角（0→-360）时建议用二阶差分；若仍有离群点，再叠加残差MAD。",
            ),
            StepParam(
                key="spike_k",
                label="剔除阈值系数(k)",
                kind="float",
                default=6.0,
                help="越小越严格；建议从6附近开始调。",
            ),
            StepParam(
                key="min_points_after_spike_filter",
                label="剔除后最少点数",
                kind="int",
                default=7,
                help="剔除后点数不足将拒绝该段。",
            ),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        spike_method_raw = str(params.get("spike_method", "二阶差分 + 残差MAD（推荐）"))
        spike_method_map = {
            # Chinese UI labels
            "二阶差分 + 残差MAD（推荐）": "diff2_then_residual",
            "仅二阶差分（适合单点尖峰）": "diff2_mad",
            "仅残差MAD（适合全局离群点）": "residual_mad",
            # Backward-compatible English values
            "diff2_then_residual": "diff2_then_residual",
            "diff2_mad": "diff2_mad",
            "residual_mad": "residual_mad",
        }
        spike_method = spike_method_map.get(spike_method_raw, "diff2_then_residual")

        cfg = _Cfg(
            lockin_channel=str(params.get("lockin_channel", "X(V)_SR-830")),
            lockin_unit=str(params.get("lockin_unit", "uV")),
            mag_jump_thresh_oe=float(params.get("mag_jump_thresh_oe", 200.0)),
            min_angle_coverage_deg=float(params.get("min_angle_coverage_deg", 330.0)),
            min_points_per_segment=int(params.get("min_points_per_segment", 30)),
            sort_angle=bool(params.get("sort_angle", True)),
            generate_plot=bool(params.get("generate_plot", True)),
            remove_spikes=bool(params.get("remove_spikes", True)),
            spike_method=spike_method,
            spike_k=float(params.get("spike_k", 6.0)),
            min_points_after_spike_filter=int(params.get("min_points_after_spike_filter", 7)),
        )

        out = StepOutputs()
        all_rows: list[dict[str, Any]] = []
        plot_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, list[float], str]] = {}

        # Generate output filenames based on uploaded input names.
        # - If multiple files share the same name, treat them as one.
        # - Otherwise join unique stems with '+'.
        seen_names: set[str] = set()
        output_stems: list[str] = []
        for filename, _ in files:
            if filename in seen_names:
                continue
            seen_names.add(filename)
            stem = os.path.splitext(os.path.basename(filename))[0]
            output_stems.append(stem)
        output_base = "+".join(output_stems) if output_stems else "ppms_fit"

        for filename, data in files:
            text = data.decode("utf-8", errors="ignore")
            headers, rows = _parse_table(text)
            if not headers or not rows:
                out.notes.append(f"{filename}: empty or unreadable.")
                continue

            mag_idx = pick_col_index(headers, ["Magnet(Oe)", "Mag Field", "Magnet", "Field"])
            angle_idx = pick_col_index(headers, ["Position (Deg)1", "Angle", "Position (Deg)", "Position"])
            lockin_idx = _pick_lockin_col(headers, cfg.lockin_channel)

            if mag_idx is None or angle_idx is None or lockin_idx is None:
                out.notes.append(f"{filename}: cannot find required columns in header.")
                continue

            mags: list[float] = []
            angles: list[float] = []
            sig_v: list[float] = []
            for r in rows:
                if max(mag_idx, angle_idx, lockin_idx) >= len(r):
                    continue
                m = _parse_float(r[mag_idx])
                a = _parse_float(r[angle_idx])
                s = _parse_float(r[lockin_idx])
                if m is None or a is None or s is None:
                    continue
                mags.append(m)
                angles.append(a)
                sig_v.append(s)

            if not mags:
                out.notes.append(f"{filename}: no numeric rows.")
                continue

            mags_arr = np.asarray(mags, dtype=float)
            ang_arr = np.asarray(angles, dtype=float)
            sig_v_arr = np.asarray(sig_v, dtype=float)

            # segment by mag jumps in original order
            seg_starts = [0]
            for i in range(1, len(mags_arr)):
                if abs(mags_arr[i] - mags_arr[i - 1]) > cfg.mag_jump_thresh_oe:
                    seg_starts.append(i)
            seg_starts.append(len(mags_arr))

            for seg_id in range(len(seg_starts) - 1):
                a0, b0 = seg_starts[seg_id], seg_starts[seg_id + 1]
                seg_mag = mags_arr[a0:b0]
                seg_ang = ang_arr[a0:b0]
                seg_sig_v = sig_v_arr[a0:b0]

                good = np.isfinite(seg_mag) & np.isfinite(seg_ang) & np.isfinite(seg_sig_v)
                seg_mag, seg_ang, seg_sig_v = seg_mag[good], seg_ang[good], seg_sig_v[good]
                if seg_mag.size == 0:
                    continue

                # If the user scanned multiple angular sweeps at the same field
                # (resetting angle back to ~0), keep only the LAST sweep.
                seg_ang, seg_mag, seg_sig_v = _pick_last_sweep_by_reset(seg_ang, seg_mag, seg_sig_v)
                # Within the last sweep, if the same angle appears multiple times,
                # keep the last occurrence (robust to small floating drift).
                seg_ang, seg_mag, seg_sig_v = _dedupe_angle_keep_last(seg_ang, seg_mag, seg_sig_v)

                mag_rep = float(np.median(seg_mag))
                ang_min = float(np.min(seg_ang))
                ang_max = float(np.max(seg_ang))
                coverage = ang_max - ang_min
                npts = int(seg_ang.size)
                npts_after_spike_filter = npts
                spike_removed_count = 0

                valid = True
                reject = ""
                if npts < cfg.min_points_per_segment:
                    valid = False
                    reject = f"points<{cfg.min_points_per_segment}"
                elif coverage < cfg.min_angle_coverage_deg:
                    valid = False
                    reject = f"coverage<{cfg.min_angle_coverage_deg}"

                if cfg.lockin_unit.lower() == "uv":
                    seg_sig_used = seg_sig_v * 1e6
                    unit = "uV"
                else:
                    seg_sig_used = seg_sig_v.copy()
                    unit = "V"

                if cfg.sort_angle:
                    idx = np.argsort(seg_ang)
                    seg_ang = seg_ang[idx]
                    seg_sig_used = seg_sig_used[idx]

                popt = [float("nan")] * 7
                r2 = float("nan")
                if valid and npts >= 7:
                    try:
                        # Initial parameter guess: A starts from 0 (user preference).
                        initial = [0.0, 0, 0, 0, 0, 0, 0]
                        # Spike filtering: for monotonic scans, diff2-based filtering is
                        # often better at removing lock-in noise spikes than global residual filtering.
                        if cfg.remove_spikes and cfg.spike_method in ("diff2_mad", "diff2_then_residual"):
                            keep_mask, removed = _spike_filter_by_diff2_mad(seg_sig_used, k=cfg.spike_k)
                            spike_removed_count += int(removed)
                            if int(np.sum(keep_mask)) >= cfg.min_points_after_spike_filter:
                                seg_ang = seg_ang[keep_mask]
                                seg_sig_used = seg_sig_used[keep_mask]
                            else:
                                valid = False
                                reject = "spike_filter_too_few_points_after_filter"

                        if not valid:
                            raise RuntimeError(reject)

                        popt_arr, _ = curve_fit(_fit_func_deg, seg_ang, seg_sig_used, p0=initial, maxfev=10000)

                        if cfg.remove_spikes and cfg.spike_method in ("residual_mad", "diff2_then_residual"):
                            keep_mask, removed = _spike_filter_by_residual_mad(seg_ang, seg_sig_used, popt_arr, k=cfg.spike_k)
                            spike_removed_count += int(removed)
                            if int(np.sum(keep_mask)) >= cfg.min_points_after_spike_filter:
                                seg_ang = seg_ang[keep_mask]
                                seg_sig_used = seg_sig_used[keep_mask]
                                popt_arr, _ = curve_fit(_fit_func_deg, seg_ang, seg_sig_used, p0=popt_arr, maxfev=10000)
                            else:
                                valid = False
                                reject = "spike_filter_too_few_points_after_filter"

                        popt = [float(x) for x in popt_arr.tolist()]
                        y_pred = _fit_func_deg(seg_ang, *popt_arr)
                        r2 = _r2(seg_sig_used, y_pred)
                    except Exception:
                        valid = False
                        reject = "curve_fit_failed"
                    finally:
                        # Update post-filter point count for reporting.
                        npts_after_spike_filter = int(seg_ang.size)

                all_rows.append(
                    {
                        "File": filename,
                        "SegmentID": seg_id,
                        "Mag_Oe": mag_rep,
                        "Angle_min": ang_min,
                        "Angle_max": ang_max,
                        "Angle_coverage": coverage,
                        "N_points": npts,
                        "Valid": bool(valid and reject == ""),
                        "RejectReason": reject,
                        "Lockin_channel": cfg.lockin_channel,
                        "Lockin_unit": unit,
                        "A": popt[0],
                        "B": popt[1],
                        "C": popt[2],
                        "D": popt[3],
                        "E": popt[4],
                        "F": popt[5],
                        "G": popt[6],
                        "R2": r2,
                        "N_points_after_spike_filter": npts_after_spike_filter,
                        "SpikeRemovedCount": spike_removed_count,
                    }
                )

                if bool(valid and reject == ""):
                    plot_series[(filename, seg_id)] = (seg_ang.copy(), seg_sig_used.copy(), popt.copy(), unit)

        if not all_rows:
            out.notes.append("No valid segments found.")
            return out

        df = pd.DataFrame(all_rows)
        out.tables["PPMS Fit Results"] = df
        out.downloads["Excel"] = (
            f"{output_base}_ppms_fit_results.xlsx",
            _to_excel_bytes(df),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if cfg.generate_plot:
            valid_df = df[df["Valid"] == True].copy()  # noqa: E712
            if len(valid_df) > 0:
                n = len(valid_df)
                cols = min(3, n)
                rows = int(math.ceil(n / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.8), squeeze=False)
                axes_flat = axes.flatten()

                for i, (_, r) in enumerate(valid_df.reset_index(drop=True).iterrows()):
                    ax = axes_flat[i]
                    mag = float(r["Mag_Oe"])
                    A, B, C, D, E, F, G = [float(r[k]) for k in ["A", "B", "C", "D", "E", "F", "G"]]
                    key = (str(r["File"]), int(r["SegmentID"]))
                    unit = str(r["Lockin_unit"])
                    if key in plot_series:
                        xs, ys, _, unit = plot_series[key]
                        ax.scatter(xs, ys, s=12, alpha=0.85)
                        x_fit = np.linspace(float(np.min(xs)), float(np.max(xs)), 300)
                    else:
                        x_fit = np.linspace(0.0, 360.0, 361)
                    y_fit = _fit_func_deg(x_fit, A, B, C, D, E, F, G)
                    ax.plot(x_fit, y_fit, "r-", lw=1.5)
                    ax.set_title(f"{r['File']} | seg={int(r['SegmentID'])} | H={mag:.0f} Oe | R2={float(r['R2']):.3f}")
                    ax.set_xlabel("Angle (deg)")
                    ax.set_ylabel(f"Signal ({unit})")
                    ax.grid(True, ls="--", alpha=0.3)

                for j in range(n, len(axes_flat)):
                    axes_flat[j].axis("off")

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                plt.close(fig)
                buf.seek(0)
                out.images["Plot"] = (f"{output_base}_ppms_fit_plot.png", buf.getvalue(), "image/png")
            else:
                out.notes.append("All segments rejected; no plot.")

        return out


step = PpmsAngleFitStep()

