from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Mapping

import pandas as pd

from ..models import StepMeta, StepOutputs, StepParam


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return h, mi


def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _combine_place_datetime(d: date, tstr: str, *, use_now: bool) -> datetime:
    if use_now:
        return datetime.now().replace(microsecond=0)
    parsed = _parse_hhmm(tstr)
    if parsed is None:
        h, mi = 0, 0
    else:
        h, mi = parsed
    return datetime.combine(d, datetime.min.time().replace(hour=h, minute=mi))


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _add_minutes(t0: datetime, minutes: float) -> datetime:
    return t0 + timedelta(minutes=float(minutes))


class Ni4MoProcessScheduleStep:
    """
    Ni4Mo 工艺时间轴：从放入时刻起按各阶段时长累加，给出两次生长区间及降温前后时刻。
    """

    meta = StepMeta(
        id="ni4mo_process_schedule",
        name="Ni4Mo 工艺时间表（生长 / 降温前后）",
        category="🕒 辅助工具",
        description=(
            "从**放入时间**起，按顺序累加各阶段时长（分钟），计算：\n\n"
            "- 两次**生长**阶段的起止时刻；\n"
            "- **降温前**（末段退火结束）与**降温后**（降温结束）时刻。\n\n"
            "勾选「使用当前时间作为放入时刻」时，以运行瞬间为起点；否则填写日期与 `HH:MM`。\n"
            "各阶段时长未填或非数字时按 **0** 处理（与参数面板中的数字一致）。\n\n"
            "**文件**：无需上传（`file_types` 为空时 RunCenter 允许直接运行）。"
        ),
        file_types=[],
        params=[
            StepParam(
                key="use_now",
                label="使用当前时间作为放入时刻",
                kind="bool",
                default=True,
                help="开启后忽略下方「放入时刻」；关闭后使用日期+时刻。",
            ),
            StepParam(
                key="place_date",
                label="放入日期 (YYYY-MM-DD，空=今天)",
                kind="str",
                default="",
            ),
            StepParam(
                key="place_time",
                label="放入时刻 (HH:MM，先输入再计算)",
                kind="str",
                default="10:30",
                help="关闭「当前时间」时有效。",
            ),
            StepParam(key="t_heat_ramp_min", label="升温时间 (分钟)", kind="float", default=10.0),
            StepParam(key="t_hold1_min", label="保温① (分钟)，默认 1h=60", kind="float", default=60.0),
            StepParam(key="t_grow1_min", label="生长① (分钟)", kind="float", default=6.0),
            StepParam(key="t_anneal1_min", label="退火① (分钟)，默认 1h=60", kind="float", default=60.0),
            StepParam(key="t_hold2_min", label="保温② (分钟)", kind="float", default=20.0),
            StepParam(key="t_grow2_min", label="生长② (分钟)", kind="float", default=39.0),
            StepParam(key="t_anneal2_min", label="退火② (分钟)，默认 1h=60", kind="float", default=60.0),
            StepParam(key="t_cool_min", label="降温时间 (分钟)", kind="float", default=20.0),
        ],
    )

    def run(self, *, files: list[tuple[str, bytes]], params: Mapping[str, Any]) -> StepOutputs:
        use_now = bool(params.get("use_now", True))
        d_raw = str(params.get("place_date", "") or "").strip()
        d = _parse_date(d_raw) or date.today()
        t0 = _combine_place_datetime(d, str(params.get("place_time", "") or ""), use_now=use_now)

        def _f(key: str) -> float:
            v = params.get(key, 0)
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        m_heat = _f("t_heat_ramp_min")
        m_h1 = _f("t_hold1_min")
        m_g1 = _f("t_grow1_min")
        m_a1 = _f("t_anneal1_min")
        m_h2 = _f("t_hold2_min")
        m_g2 = _f("t_grow2_min")
        m_a2 = _f("t_anneal2_min")
        m_cool = _f("t_cool_min")

        cur = t0
        rows_phases: list[dict[str, Any]] = []

        def seg(name: str, minutes: float) -> None:
            nonlocal cur
            start = cur
            cur = _add_minutes(cur, minutes)
            rows_phases.append(
                {
                    "阶段": name,
                    "时长_min": minutes,
                    "开始": _fmt(start),
                    "结束": _fmt(cur),
                }
            )

        seg("升温", m_heat)
        seg("保温①", m_h1)
        g1_start = cur
        seg("生长①", m_g1)
        g1_end = cur
        seg("退火①", m_a1)
        seg("保温②", m_h2)
        g2_start = cur
        seg("生长②", m_g2)
        g2_end = cur
        seg("退火②", m_a2)
        before_cool = cur
        seg("降温", m_cool)
        after_cool = cur

        df_phases = pd.DataFrame(rows_phases)

        df_four = pd.DataFrame(
            [
                {"序号": 1, "项": "第一次生长（段起始时刻）", "时刻": _fmt(g1_start)},
                {"序号": 2, "项": "第二次生长（段起始时刻）", "时刻": _fmt(g2_start)},
                {"序号": 3, "项": "降温前（末段退火结束）", "时刻": _fmt(before_cool)},
                {"序号": 4, "项": "降温后（降温结束）", "时刻": _fmt(after_cool)},
            ]
        )
        df_grow_windows = pd.DataFrame(
            [
                {"生长段": "生长①", "开始": _fmt(g1_start), "结束": _fmt(g1_end), "时长_min": m_g1},
                {"生长段": "生长②", "开始": _fmt(g2_start), "结束": _fmt(g2_end), "时长_min": m_g2},
            ]
        )

        out = StepOutputs()
        out.tables["四个关键时间（自放入起累加）"] = df_four
        out.tables["生长阶段区间"] = df_grow_windows
        out.tables["全流程累加明细"] = df_phases
        out.notes.append(f"放入基准：{_fmt(t0)}（自该时刻起累加各阶段分钟数）。")
        return out


step = Ni4MoProcessScheduleStep()
