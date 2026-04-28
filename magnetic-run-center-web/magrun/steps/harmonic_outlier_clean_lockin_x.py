from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..models import StepMeta, StepOutputs, StepParam
from ..utils.text_parse import _norm, pick_col_index, split_columns


def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            safe = name[:31] if name else "sheet"
            df.to_excel(w, index=False, sheet_name=safe)
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


def _find_header_idx(lines: list[str], *, required_cols: list[list[str]]) -> int:
    """
    Find a header row by looking for groups of required columns.
    Falls back to first non-empty line if not found.
    """
    for i, ln in enumerate(lines):
        cols = split_columns(ln)
        if not cols:
            continue
        cols_norm = [_norm(c) for c in cols]
        ok = True
        for group in required_cols:
            gnorm = [_norm(x) for x in group]
            if not any(any(g in c for c in cols_norm) for g in gnorm):
                ok = False
                break
        if ok:
            return i
    return 0


@dataclass(frozen=True)
class _Cfg:
    mag_tol_oe: float
    z_thresh: float
    hard_dev_v: float
    min_points_per_segment: int
    x_col: str


def _segment_by_mag_tolerance(mag: np.ndarray, *, tol_oe: float) -> np.ndarray:
    """
    Segment points (in original order) into magnet "platforms" by tolerance.
    A new segment starts when |mag - current_center| > tol_oe.
    Segment center is updated as a running mean of assigned points.
    """
    n = int(mag.size)
    if n == 0:
        return np.zeros(0, dtype=int)
    seg = np.zeros(n, dtype=int)
    cur = 0
    center = float(mag[0])
    count = 1
    for i in range(1, n):
        mi = float(mag[i])
        if not np.isfinite(mi):
            seg[i] = cur
            continue
        if abs(mi - center) <= tol_oe:
            seg[i] = cur
            count += 1
            center = center + (mi - center) / float(count)
        else:
            cur += 1
            seg[i] = cur
            center = mi
            count = 1
    return seg


def _robust_outlier_mask(
    x: np.ndarray,
    *,
    z_thresh: float,
    hard_dev: float,
) -> tuple[np.ndarray, float, float]:
    """
    Return keep_mask (True=keep), plus (median, mad).
    Policy: keep if both
    - robust z-score <= z_thresh, where rz = |x-med| / (1.4826*MAD)
    - absolute deviation <= hard_dev
    NaNs are always dropped.
    """
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x, dtype=bool), float("nan"), float("nan")
    med = float(np.nanmedian(x[good]))
    dev = np.abs(x - med)
    mad = float(np.nanmedian(dev[good]))

    keep = np.isfinite(x)
    keep &= dev <= hard_dev
    if mad > 0 and np.isfinite(mad):
        rz = dev / (1.4826 * mad)
        keep &= rz <= z_thresh
    # else: MAD=0 -> only hard_dev (already applied)
    return keep, med, mad


class HarmonicOutlierCleanLockinXStep:
    meta = StepMeta(
        id="harmonic_outlier_clean_lockin_x",
        name="Harmonic: 跳点清洗（按磁场分段，Lock-in X）",
        category="📊 谐波数据处理",
        description=(
            "对原始表格中的 Lock-in 信号列（默认 `X(V)_SR-830-1`）做跳点/离群点删除，并且**按磁场分段**分别判断（不同磁场段不是同一群）。\n\n"
            "分段逻辑：按行顺序用 `Magnet(Oe)` 进行平台聚类：若 |Magnet - 当前段中心| ≤ mag_tol_oe，则归入当前段，否则开启新段。\n\n"
            "每段离群规则（只删极端点）：\n"
            "- 先计算段内 median 与 MAD；\n"
            "- 保留条件同时满足：\n"
            "  - |x - median| ≤ hard_dev_v\n"
            "  - robust_z = |x - median| / (1.4826*MAD) ≤ z_thresh（MAD=0 时仅用 hard_dev_v）\n\n"
            "输出：清洗后的同列同表（删行）+ 统计表 + 被删点清单（含磁场段信息）。"
        ),
        file_types=["txt"],
        params=[
            StepParam(key="mag_tol_oe", label="磁场平台容差 mag_tol_oe (Oe)", kind="float", default=80.0),
            StepParam(key="z_thresh", label="鲁棒 z 阈值 z_thresh（越大越保守）", kind="float", default=10.0),
            StepParam(key="hard_dev_v", label="硬阈值 |x-median| ≤ hard_dev_v (V)", kind="float", default=1e-6),
            StepParam(key="min_points_per_segment", label="段内最少点数（不足则不删）", kind="int", default=20),
            StepParam(
                key="x_col",
                label="信号列（默认 X(V)_SR-830-1）",
                kind="select",
                default="X(V)_SR-830-1",
                options=["X(V)_SR-830-1", "X(V)_SR-830"],
                help="用于清洗的信号列名。一般选择 X(V)_SR-830-1。",
            ),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        cfg = _Cfg(
            mag_tol_oe=float(params.get("mag_tol_oe", 80.0)),
            z_thresh=float(params.get("z_thresh", 10.0)),
            hard_dev_v=float(params.get("hard_dev_v", 1e-6)),
            min_points_per_segment=int(params.get("min_points_per_segment", 20)),
            x_col=str(params.get("x_col", "X(V)_SR-830-1")),
        )

        out = StepOutputs()
        if not files:
            out.notes.append("未上传文件。")
            return out

        seg_rows: list[dict[str, Any]] = []
        drop_rows: list[dict[str, Any]] = []
        cleaned_payloads: list[tuple[str, bytes]] = []

        for fname, payload in files:
            text = payload.decode("utf-8", errors="ignore")
            raw_lines = [ln for ln in text.splitlines() if ln.strip()]
            if not raw_lines:
                out.notes.append(f"{fname}: 文件为空或无法解析。")
                continue

            header_idx = _find_header_idx(
                raw_lines,
                required_cols=[
                    ["magnet(oe)", "mag field", "magnet", "field"],
                    [_norm(cfg.x_col), "x(v)_sr-830-1", "x(v)_sr-830"],
                ],
            )
            header = split_columns(raw_lines[header_idx])
            data_lines = raw_lines[header_idx + 1 :]
            rows = [split_columns(ln) for ln in data_lines if ln.strip()]

            mag_idx = pick_col_index(header, ["Magnet(Oe)", "Mag Field", "Magnet", "Field"])
            x_idx = pick_col_index(header, [cfg.x_col])

            if mag_idx is None or x_idx is None:
                out.notes.append(f"{fname}: 找不到必要列 Magnet(Oe) 或 {cfg.x_col}，已跳过。")
                continue

            mags: list[float] = []
            xs: list[float] = []
            for r in rows:
                if max(mag_idx, x_idx) >= len(r):
                    mags.append(float("nan"))
                    xs.append(float("nan"))
                    continue
                m = _parse_float(r[mag_idx])
                x = _parse_float(r[x_idx])
                mags.append(float("nan") if m is None else float(m))
                xs.append(float("nan") if x is None else float(x))

            mag_arr = np.asarray(mags, dtype=float)
            x_arr = np.asarray(xs, dtype=float)

            seg_id = _segment_by_mag_tolerance(mag_arr, tol_oe=cfg.mag_tol_oe)
            keep = np.ones(len(rows), dtype=bool)

            for sid in np.unique(seg_id):
                idx = np.where(seg_id == sid)[0]
                seg_x = x_arr[idx]
                seg_mag = mag_arr[idx]

                n_total = int(idx.size)
                n_finite = int(np.sum(np.isfinite(seg_x)))
                mag_med = float(np.nanmedian(seg_mag)) if np.any(np.isfinite(seg_mag)) else float("nan")

                if n_finite < cfg.min_points_per_segment:
                    seg_rows.append(
                        {
                            "File": fname,
                            "SegmentID": int(sid),
                            "Magnet_median_Oe": mag_med,
                            "N_total": n_total,
                            "N_numeric": n_finite,
                            "N_dropped": 0,
                            "Drop_rate": 0.0,
                            "X_median": float(np.nanmedian(seg_x)) if n_finite else float("nan"),
                            "MAD": float("nan"),
                            "Rule": f"skip(n_numeric<{cfg.min_points_per_segment})",
                        }
                    )
                    continue

                seg_keep, seg_med, seg_mad = _robust_outlier_mask(
                    seg_x,
                    z_thresh=cfg.z_thresh,
                    hard_dev=cfg.hard_dev_v,
                )
                keep[idx] = seg_keep

                n_drop = int(np.sum(~seg_keep))
                seg_rows.append(
                    {
                        "File": fname,
                        "SegmentID": int(sid),
                        "Magnet_median_Oe": mag_med,
                        "N_total": n_total,
                        "N_numeric": n_finite,
                        "N_dropped": n_drop,
                        "Drop_rate": (n_drop / n_total) if n_total else float("nan"),
                        "X_median": seg_med,
                        "MAD": seg_mad,
                        "Rule": f"dev<={cfg.hard_dev_v:g} & rz<={cfg.z_thresh:g}",
                    }
                )

                if n_drop:
                    for local_i, is_keep in zip(idx.tolist(), seg_keep.tolist(), strict=False):
                        if is_keep:
                            continue
                        r = rows[local_i]
                        drop_rows.append(
                            {
                                "File": fname,
                                "SegmentID": int(sid),
                                "Magnet_Oe": mag_arr[local_i],
                                "X": x_arr[local_i],
                                "RowIndex_in_data": int(local_i),
                                "RowText": "\t".join(r),
                            }
                        )

            kept_rows = [r for i, r in enumerate(rows) if bool(keep[i])]
            cleaned_lines = ["\t".join(header)] + ["\t".join(r) for r in kept_rows]
            cleaned_text = "\n".join(cleaned_lines) + "\n"
            cleaned_name = fname.rsplit(".", 1)[0] + "_cleaned.txt"
            cleaned_payloads.append((cleaned_name, cleaned_text.encode("utf-8")))

            out.notes.append(
                f"{fname}: 行数 {len(rows)} -> {len(kept_rows)}（删 {len(rows) - len(kept_rows)}）; "
                f"segments={int(np.max(seg_id) + 1) if len(seg_id) else 0}."
            )

        if not cleaned_payloads:
            out.notes.append("未成功处理任何文件（可能缺列或为空）。")
            return out

        seg_df = pd.DataFrame(seg_rows).sort_values(["File", "SegmentID"], kind="stable") if seg_rows else pd.DataFrame()
        drop_df = pd.DataFrame(drop_rows).sort_values(["File", "SegmentID", "RowIndex_in_data"], kind="stable") if drop_rows else pd.DataFrame()

        if len(files) == 1:
            out.downloads["Cleaned"] = (cleaned_payloads[0][0], cleaned_payloads[0][1], "text/plain")
        else:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in cleaned_payloads:
                    zf.writestr(name, data)
            zbuf.seek(0)
            out.downloads["ZIP"] = ("cleaned_files.zip", zbuf.getvalue(), "application/zip")

        sheets: dict[str, pd.DataFrame] = {}
        if len(seg_df):
            sheets["segments"] = seg_df
            out.tables["Segments summary"] = seg_df
        if len(drop_df):
            sheets["dropped_points"] = drop_df
            out.tables["Dropped points"] = drop_df.head(2000)
        if sheets:
            out.downloads["Excel"] = (
                "outlier_clean_report.xlsx",
                _to_excel_bytes(sheets),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if not len(drop_df):
            out.notes.append("未检测到离群点（按当前阈值）。")
        else:
            out.notes.append(f"共删除离群点 {len(drop_df)} 行（所有文件合计）。")

        return out


step = HarmonicOutlierCleanLockinXStep()

