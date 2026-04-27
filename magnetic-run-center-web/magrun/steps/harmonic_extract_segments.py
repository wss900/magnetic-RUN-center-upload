from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import StepMeta, StepOutputs, StepParam


def _parse_float_first_col(line: str) -> float | None:
    parts = line.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except Exception:
        return None


def _extract_segments_from_text(text: str, *, range_min: float, range_max: float, tolerance: float) -> tuple[str | None, str | None]:
    """
    Extract ascending and descending segments from a scan text.

    Logic matches the original script:
    - Ascending: start when first col ~= range_min, stop when first col ~= range_max
    - Descending: after ascending end, start when first col ~= range_max, stop when first col ~= range_min
    - Comparison uses abs(val - target) < tolerance
    - Lines are kept verbatim (all columns preserved)
    """

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None

    asc: list[str] = []
    found_min = False
    found_max = False
    for ln in lines:
        val = _parse_float_first_col(ln)
        if val is None:
            continue
        if not found_min:
            if abs(val - range_min) < tolerance:
                found_min = True
                asc.append(ln)
            continue
        asc.append(ln)
        if abs(val - range_max) < tolerance:
            found_max = True
            break

    if not found_min or not found_max:
        return None, None

    # locate where ascending ended in original lines
    try:
        after_idx = lines.index(asc[-1]) + 1
    except Exception:
        after_idx = len(lines)

    desc: list[str] = []
    found_max2 = False
    found_min2 = False
    for ln in lines[after_idx:]:
        val = _parse_float_first_col(ln)
        if val is None:
            continue
        if not found_max2:
            if abs(val - range_max) < tolerance:
                found_max2 = True
                desc.append(ln)
            continue
        desc.append(ln)
        if abs(val - range_min) < tolerance:
            found_min2 = True
            break

    if not found_max2 or not found_min2:
        return None, None

    return "\n".join(asc), "\n".join(desc)


@dataclass(frozen=True)
class _Cfg:
    range_min: float
    range_max: float
    tolerance: float


class HarmonicExtractSegmentsStep:
    meta = StepMeta(
        id="harmonic_extract_segments",
        name="谐波提取上升/下降段",
        category="📊 谐波数据处理",
        description=(
            "上传一个扫描文件（txt），以第一列为判断标准提取两段：\n\n"
            "- 上升段：从 first_col≈range_min 开始，到 first_col≈range_max 结束\n"
            "- 下降段：从上升段结束后开始，从 first_col≈range_max 到 first_col≈range_min\n\n"
            "结果打包为 ZIP：ascending.txt + descending.txt"
        ),
        file_types=["txt"],
        params=[
            StepParam(key="range_min", label="范围下限 range_min", kind="float", default=-0.5),
            StepParam(key="range_max", label="范围上限 range_max", kind="float", default=0.5),
            StepParam(key="tolerance", label="浮点容差 tolerance", kind="float", default=1e-6),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        cfg = _Cfg(
            range_min=float(params.get("range_min", -0.5)),
            range_max=float(params.get("range_max", 0.5)),
            tolerance=float(params.get("tolerance", 1e-6)),
        )

        out = StepOutputs()
        if not files:
            out.notes.append("未上传文件。")
            return out

        if len(files) > 1:
            out.notes.append("检测到多文件上传：该功能只处理第一个文件，其它文件将被忽略。")

        filename, payload = files[0]
        text = payload.decode("utf-8", errors="ignore")
        asc, desc = _extract_segments_from_text(
            text,
            range_min=cfg.range_min,
            range_max=cfg.range_max,
            tolerance=cfg.tolerance,
        )

        if asc is None or desc is None:
            out.notes.append("提取失败：未找到完整的上升段或下降段，请检查 range_min/range_max/tolerance 或文件格式。")
            return out

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ascending.txt", asc)
            zf.writestr("descending.txt", desc)
        zip_buf.seek(0)

        out.downloads["ZIP"] = ("segments.zip", zip_buf.getvalue(), "application/zip")
        out.notes.append(f"提取完成：ascending {len(asc.splitlines())} 行，descending {len(desc.splitlines())} 行。")
        out.notes.append(f"来源文件：{filename}")
        return out


step = HarmonicExtractSegmentsStep()

