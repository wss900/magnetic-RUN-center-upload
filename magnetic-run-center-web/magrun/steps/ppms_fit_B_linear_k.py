from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models import StepMeta, StepOutputs, StepParam


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
    buf.seek(0)
    return buf.getvalue()


def _extract_current_ma(filename: str) -> float:
    m = re.search(r"(\\d+\\.?\\d*)mA", filename)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_linear_with_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # y = kx + b
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k, b = np.polyfit(x, y, 1)
    return float(k), float(b)


def _fit_linear_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # y = kx (b forced to 0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return float("nan"), 0.0
    k = float(np.sum(x * y) / denom)
    return k, 0.0


@dataclass(frozen=True)
class _Cfg:
    fit_mode: str
    min_points_per_current: int
    filter_valid_only: bool
    generate_plots: bool


class PpmsFitBLinearKStep:
    meta = StepMeta(
        id="ppms_fit_B_linear_k",
        name="对拟合后的 B 列线性拟合（按电流求 K）",
        category="🧪 PPMS 数据处理",
        description=(
            "读取上一步 `PPMS 角度扫描拟合（多段磁场）` 导出的 Excel（含 `PPMS Fit Results` 表）。\n\n"
            "对每个电流（从 `File` 字段的文件名中提取 `xxmA`）分别做线性拟合：\n"
            "- x 轴：七参数拟合后的 B\n"
            "- y 轴：1 / (Mag_Oe/1000 + 1)\n"
            "每个电流得到一个 K（斜率），并生成对应散点+拟合线图片，便于检查。"
        ),
        file_types=["xlsx"],
        params=[
            StepParam(
                key="fit_mode",
                label="线性拟合方式",
                kind="select",
                default="带截距：y = kx + b",
                options=["带截距：y = kx + b", "过原点：y = kx"],
            ),
            StepParam(key="min_points_per_current", label="每个电流最少点数", kind="int", default=3),
            StepParam(
                key="filter_valid_only",
                label="仅使用 Valid=True 的点（若存在该列）",
                kind="bool",
                default=True,
            ),
            StepParam(key="generate_plots", label="生成每个电流的拟合图（PNG）", kind="bool", default=True),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        cfg = _Cfg(
            fit_mode=str(params.get("fit_mode", "带截距：y = kx + b")),
            min_points_per_current=int(params.get("min_points_per_current", 3)),
            filter_valid_only=bool(params.get("filter_valid_only", True)),
            generate_plots=bool(params.get("generate_plots", True)),
        )

        out = StepOutputs()

        # Output base name from input files.
        stems: list[str] = []
        seen: set[str] = set()
        for fname, _ in files:
            if fname in seen:
                continue
            seen.add(fname)
            stems.append(os.path.splitext(os.path.basename(fname))[0])
        output_base = "+".join(stems) if stems else "B_linear_fit"

        # Load and combine input excel(s)
        dfs: list[pd.DataFrame] = []
        for fname, payload in files:
            try:
                bio = io.BytesIO(payload)
                # Prefer sheet "PPMS Fit Results" if present, else first sheet.
                xls = pd.ExcelFile(bio)
                sheet = "PPMS Fit Results" if "PPMS Fit Results" in xls.sheet_names else xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
                df["__source_excel__"] = fname
                dfs.append(df)
            except Exception:
                out.notes.append(f"{fname}: 无法读取 Excel。请确认是上一步导出的 .xlsx。")

        if not dfs:
            out.notes.append("No readable Excel files.")
            return out

        data = pd.concat(dfs, ignore_index=True)

        # Basic column checks
        required = ["File", "Mag_Oe", "B"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            out.notes.append(f"Missing required columns: {missing}. Expecting PPMS Fit Results from previous step.")
            return out

        if cfg.filter_valid_only and "Valid" in data.columns:
            data = data[data["Valid"] == True].copy()  # noqa: E712

        # Extract current from filename
        data["current_mA"] = data["File"].astype(str).map(_extract_current_ma)
        data = data[np.isfinite(data["current_mA"])].copy()
        if len(data) == 0:
            out.notes.append("未从 `File` 字段提取到电流（xxmA）。请确认文件名包含例如 `15mA`。")
            return out

        # Build x/y
        data["x_B"] = pd.to_numeric(data["B"], errors="coerce")
        data["H_kOe"] = pd.to_numeric(data["Mag_Oe"], errors="coerce") / 1000.0
        data["y_inv"] = 1.0 / (data["H_kOe"] + 1.0)
        data = data[np.isfinite(data["x_B"]) & np.isfinite(data["y_inv"])].copy()
        if len(data) == 0:
            out.notes.append("No finite rows after computing x/y.")
            return out

        rows_out: list[dict[str, Any]] = []

        fit_through_origin = "过原点" in cfg.fit_mode

        for current, g in data.groupby("current_mA", sort=True):
            x = g["x_B"].to_numpy(dtype=float)
            y = g["y_inv"].to_numpy(dtype=float)
            n = int(x.size)

            status = "ok"
            reason = ""
            k = float("nan")
            b = float("nan")
            r2 = float("nan")

            if n < cfg.min_points_per_current:
                status = "rejected"
                reason = f"not_enough_points<{cfg.min_points_per_current}"
            else:
                try:
                    if fit_through_origin:
                        k, b = _fit_linear_through_origin(x, y)
                    else:
                        k, b = _fit_linear_with_intercept(x, y)
                    y_pred = k * x + b
                    r2 = _r2(y, y_pred)
                except Exception:
                    status = "failed"
                    reason = "linear_fit_failed"

            rows_out.append(
                {
                    "current_mA": float(current),
                    "N_points": n,
                    "k": float(k),
                    "b": float(b) if not fit_through_origin else 0.0,
                    "R2": float(r2),
                    "fit_mode": cfg.fit_mode,
                    "status": status,
                    "reason": reason,
                }
            )

            if cfg.generate_plots and status == "ok":
                fig, ax = plt.subplots(figsize=(7.6, 5.6))
                ax.scatter(x, y, s=20, alpha=0.85)
                xfit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                yfit = k * xfit + (0.0 if fit_through_origin else b)
                ax.plot(xfit, yfit, "r-", lw=1.8)

                ax.set_xlabel("B (from 7-parameter fit)")
                ax.set_ylabel("1 / (Mag_Oe/1000 + 1)")
                ax.set_title(f"{float(current):g} mA | k={k:.6g} | R2={r2:.3f} | N={n}")
                ax.grid(True, ls="--", alpha=0.3)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                plt.close(fig)
                buf.seek(0)

                img_name = f"{output_base}_Kfit_{float(current):g}mA.png"
                out.images[f"{float(current):g} mA"] = (img_name, buf.getvalue(), "image/png")

        res_df = pd.DataFrame(rows_out).sort_values("current_mA").reset_index(drop=True)
        out.tables["K results"] = res_df
        # Point-level details (for downstream processing / auditing)
        details_cols = [
            "current_mA",
            "File",
            "SegmentID",
            "Mag_Oe",
            "B",
            "x_B",
            "y_inv",
            "H_kOe",
        ]
        details_present = [c for c in details_cols if c in data.columns]
        details_df = data[details_present].copy()
        out.tables["Point details (B vs y)"] = details_df.sort_values(["current_mA"]).reset_index(drop=True)
        out.downloads["Excel"] = (
            f"{output_base}_B_linear_fit_K.xlsx",
            _to_excel_bytes(res_df),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        out.downloads["Point details Excel"] = (
            f"{output_base}_B_linear_fit_point_details.xlsx",
            _to_excel_bytes(details_df),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        return out


step = PpmsFitBLinearKStep()

