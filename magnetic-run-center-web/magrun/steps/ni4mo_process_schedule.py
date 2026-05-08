from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Mapping

import pandas as pd

from ..models import StepMeta, StepOutputs, StepParam


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    """
    只解析「时:分」或「时.分」，不要日期。
    支持：10:30、10：30、10.30、10:30:00（秒忽略，仍按分对齐）。
    """
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace("：", ":").replace("．", ".").strip()
    m = re.match(r"^(\d{1,2})[:.](\d{2})$", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    else:
        m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", s)
        if not m:
            return None
        h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return h, mi


def _baseline_datetime(*, place_time: str, use_now: bool) -> datetime:
    """放入基准：当前时间，或「今天 + 指定时刻」（不输入日期）。"""
    if use_now:
        return datetime.now().replace(microsecond=0)
    d = date.today()
    parsed = _parse_hhmm(place_time)
    if parsed is None:
        h, mi = 0, 0
    else:
        h, mi = parsed
    return datetime.combine(d, datetime.min.time().replace(hour=h, minute=mi))


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _add_minutes(t0: datetime, minutes: float) -> datetime:
    return t0 + timedelta(minutes=float(minutes))


def _build_phone_alarm_prompt(
    *,
    g1_start: datetime,
    g2_start: datetime,
    before_cool: datetime,
    after_cool: datetime,
) -> str:
    """一段可复制给手机 AI 的自然语言口令（含四个时刻说明）。"""
    lines = [
        "帮我在手机里增加下面 4 个闹钟。每个闹钟的标签/备注请用括号里的说明，时间用后面的日期时间（到点响铃或强提醒即可）：",
        "",
        f"1. 第一次生长（段起始时刻）— {_fmt(g1_start)}",
        f"2. 第二次生长（段起始时刻）— {_fmt(g2_start)}",
        f"3. 降温前（末段退火结束）— {_fmt(before_cool)}",
        f"4. 降温后（降温结束）— {_fmt(after_cool)}",
    ]
    return "\n".join(lines)


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
            "勾选「使用当前时间作为放入时刻」时，以运行瞬间为起点；否则**只填时刻、不填日期**，基准日为**运行当天**，例如 `10:30` 或 `10.30`。\n"
            "各阶段时长未填或非数字时按 **0** 处理（与参数面板中的数字一致）。\n\n"
            "**文件**：无需上传（`file_types` 为空时 RunCenter 允许直接运行）。\n\n"
            "运行后会给出**可复制给手机 AI 的闹钟口令**（表格 + 说明区），并可下载 `ni4mo_手机闹钟口令.txt` 便于长按复制。"
        ),
        file_types=[],
        params=[
            StepParam(
                key="use_now",
                label="使用当前时间作为放入时刻",
                kind="bool",
                default=True,
                help="关闭后使用下方「放入时刻」；基准日为今天，无需填日期。",
            ),
            StepParam(
                key="place_time",
                label="放入时刻（仅时刻，如 10:30 或 10.30）",
                kind="str",
                default="10:30",
                help="关闭「当前时间」时有效；支持冒号或英文句点分隔。留空或无法识别时按 0:00。",
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
        place_time_raw = str(params.get("place_time", "") or "")
        t0 = _baseline_datetime(place_time=place_time_raw, use_now=use_now)

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

        alarm_prompt = _build_phone_alarm_prompt(
            g1_start=g1_start,
            g2_start=g2_start,
            before_cool=before_cool,
            after_cool=after_cool,
        )
        prompt_bytes = alarm_prompt.encode("utf-8")

        out = StepOutputs()
        out.tables["四个关键时间（自放入起累加）"] = df_four
        out.tables["生长阶段区间"] = df_grow_windows
        out.tables["全流程累加明细"] = df_phases
        out.tables["复制给手机AI（闹钟口令）"] = pd.DataFrame([{"全文": alarm_prompt}])
        out.notes.append(
            f"放入基准：{_fmt(t0)}（自该时刻起累加各阶段分钟数；"
            + ("运行时刻" if use_now else f"今天 {date.today().isoformat()} + 填入时刻")
            + "）。"
        )
        if (not use_now) and place_time_raw.strip() and (_parse_hhmm(place_time_raw) is None):
            out.notes.append("提示：未能识别「放入时刻」格式，已按当天 00:00 作为基准。")
        out.notes.append("【复制】下面整段可发给手机里的 AI，用于添加闹钟：\n\n" + alarm_prompt)
        out.downloads["手机闹钟口令 txt"] = ("ni4mo_手机闹钟口令.txt", prompt_bytes, "text/plain; charset=utf-8")
        return out


step = Ni4MoProcessScheduleStep()
