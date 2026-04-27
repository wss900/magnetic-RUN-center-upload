from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..models import StepMeta, StepOutputs, StepParam
from ..utils.text_parse import parse_3col_numeric_table


def _to_float(x: str) -> float | None:
    s = x.strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _quad(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x**2 + b * x + c


def _lin(x: np.ndarray, m: float, c: float) -> np.ndarray:
    return m * x + c


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
    buf.seek(0)
    return buf.getvalue()


@dataclass(frozen=True)
class _Cfg:
    h_min: float
    h_max: float
    min_points: int
    extract_current_from_filename: bool
    generate_plot: bool


class HarmonicSlopeCurvatureStep:
    meta = StepMeta(
        id="harmonic_slope_curvature",
        name="谐波斜率曲率分析（First 二次拟合 + Second 线性拟合）",
        category="📊 谐波数据处理",
        description=(
            "输入三列：H(kOe), First harmonic(V), Second harmonic(V)。\n\n"
            "- 在选择的 H 范围内：对 First harmonic 做二次拟合，取二次项系数 a 作为 B2_curvature；\n"
            "- 对 Second harmonic 做线性拟合，取斜率 m 作为 linear_slope；\n"
            "- ratio = -m / a。\n\n"
            "如文件名包含 `xxmA`，会提取电流并绘制 ratio vs current 图。"
        ),
        file_types=["txt"],
        params=[
            StepParam(key="H_min", label="拟合范围 H_min (kOe)", kind="float", default=-0.5),
            StepParam(key="H_max", label="拟合范围 H_max (kOe)", kind="float", default=0.5),
            StepParam(key="min_points", label="区间内最少点数", kind="int", default=10),
            StepParam(
                key="extract_current_from_filename",
                label="从文件名提取电流 (xxmA)",
                kind="bool",
                default=True,
            ),
            StepParam(key="generate_plot", label="生成 ratio-current 图", kind="bool", default=True),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        cfg = _Cfg(
            h_min=float(params.get("H_min", -0.5)),
            h_max=float(params.get("H_max", 0.5)),
            min_points=int(params.get("min_points", 10)),
            extract_current_from_filename=bool(params.get("extract_current_from_filename", True)),
            generate_plot=bool(params.get("generate_plot", True)),
        )

        out = StepOutputs()
        rows_out: list[dict[str, Any]] = []

        for fname, payload in files:
            text = payload.decode("utf-8", errors="ignore")
            headers, rows = parse_3col_numeric_table(text)

            H: list[float] = []
            first: list[float] = []
            second: list[float] = []
            for r in rows:
                if len(r) < 3:
                    continue
                h = _to_float(r[0])
                f1 = _to_float(r[1])
                f2 = _to_float(r[2])
                if h is None or f1 is None or f2 is None:
                    continue
                H.append(h)
                first.append(f1)
                second.append(f2)

            status = "ok"
            reason = ""
            current_ma = float("nan")

            if cfg.extract_current_from_filename:
                m = re.search(r"(\\d+\\.?\\d*)mA", fname)
                if m:
                    try:
                        current_ma = float(m.group(1))
                    except Exception:
                        current_ma = float("nan")

            if len(H) == 0:
                status = "failed"
                reason = "no_numeric_rows"
                rows_out.append(
                    {
                        "filename": fname,
                        "current_mA": current_ma,
                        "H_min": cfg.h_min,
                        "H_max": cfg.h_max,
                        "N_points_used": 0,
                        "B2_curvature": float("nan"),
                        "linear_slope": float("nan"),
                        "ratio": float("nan"),
                        "status": status,
                        "reason": reason,
                    }
                )
                continue

            H_arr = np.asarray(H, dtype=float)
            f1_arr = np.asarray(first, dtype=float)
            f2_arr = np.asarray(second, dtype=float)

            mask = (H_arr >= cfg.h_min) & (H_arr <= cfg.h_max) & np.isfinite(f1_arr) & np.isfinite(f2_arr)
            x = H_arr[mask]
            y1 = f1_arr[mask]
            y2 = f2_arr[mask]

            n_used = int(x.size)
            if n_used < cfg.min_points:
                status = "rejected"
                reason = f"not_enough_points<{cfg.min_points}"
                rows_out.append(
                    {
                        "filename": fname,
                        "current_mA": current_ma,
                        "H_min": cfg.h_min,
                        "H_max": cfg.h_max,
                        "N_points_used": n_used,
                        "B2_curvature": float("nan"),
                        "linear_slope": float("nan"),
                        "ratio": float("nan"),
                        "status": status,
                        "reason": reason,
                    }
                )
                continue

            B2 = float("nan")
            slope = float("nan")

            try:
                popt_q, _ = curve_fit(_quad, x, y1, maxfev=10000)
                B2 = float(popt_q[0])
            except Exception:
                status = "failed"
                reason = "quad_fit_failed"

            try:
                popt_l, _ = curve_fit(_lin, x, y2, maxfev=10000)
                slope = float(popt_l[0])
            except Exception:
                status = "failed"
                reason = "lin_fit_failed" if not reason else reason + ";lin_fit_failed"

            ratio = float("nan")
            if np.isfinite(B2) and np.isfinite(slope) and B2 != 0:
                ratio = float(-slope / B2)
            elif status == "ok":
                status = "failed"
                reason = "invalid_B2_or_slope"

            rows_out.append(
                {
                    "filename": fname,
                    "current_mA": current_ma,
                    "H_min": cfg.h_min,
                    "H_max": cfg.h_max,
                    "N_points_used": n_used,
                    "B2_curvature": B2,
                    "linear_slope": slope,
                    "ratio": ratio,
                    "status": status,
                    "reason": reason,
                }
            )

        if not rows_out:
            out.notes.append("No files processed.")
            return out

        df = pd.DataFrame(rows_out)
        out.tables["Harmonic slope/curvature"] = df
        out.downloads["Excel"] = (
            "harmonic_slope_curvature.xlsx",
            _to_excel_bytes(df),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if cfg.generate_plot:
            plot_df = df[np.isfinite(df["current_mA"]) & np.isfinite(df["ratio"])].copy()
            if len(plot_df) >= 1:
                mA = plot_df["current_mA"].to_numpy(dtype=float)
                ratio = plot_df["ratio"].to_numpy(dtype=float)
                sort_idx = np.argsort(mA)
                mA_s = mA[sort_idx]
                ratio_s = ratio[sort_idx]

                fig, ax = plt.subplots(figsize=(8, 6))
                pos = ratio_s > 0
                neg = ratio_s < 0

                if np.any(pos):
                    xp = mA_s[pos]
                    yp = ratio_s[pos]
                    ax.scatter(xp, yp, c="red", label="Positive")
                    if len(xp) >= 2 and float(np.sum(xp * xp)) != 0.0:
                        kp = float(np.sum(xp * yp) / np.sum(xp * xp))
                        xfit = np.linspace(0, float(np.max(xp)), 50)
                        ax.plot(xfit, kp * xfit, "r--", label=f"k+={kp:.4g}")

                if np.any(neg):
                    xn = mA_s[neg]
                    yn = ratio_s[neg]
                    ax.scatter(xn, yn, c="blue", label="Negative")
                    if len(xn) >= 2 and float(np.sum(xn * xn)) != 0.0:
                        kn = float(np.sum(xn * yn) / np.sum(xn * xn))
                        xfit = np.linspace(0, float(np.max(xn)), 50)
                        ax.plot(xfit, kn * xfit, "b--", label=f"k-={kn:.4g}")

                ax.axhline(0, color="grey", lw=1)
                ax.axvline(0, color="grey", lw=1)
                ax.set_xlabel("Current (mA)")
                ax.set_ylabel("ratio = -slope / B2")
                ax.set_title("Ratio vs Current")
                ax.grid(True, ls="--", alpha=0.3)
                ax.legend()

                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=150)
                plt.close(fig)
                buf.seek(0)
                out.images["Plot"] = ("harmonic_ratio_vs_current.png", buf.getvalue(), "image/png")
            else:
                out.notes.append("无法生成 ratio-current 图：文件名未提取到 mA 或 ratio 全为 NaN。")

        return out


step = HarmonicSlopeCurvatureStep()

