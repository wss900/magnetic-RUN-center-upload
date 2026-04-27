from __future__ import annotations

import streamlit as st

from magrun.runner import load_steps


st.set_page_config(page_title="磁性测量数据处理平台", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
    .main-title { font-size: 2.0rem; font-weight: 800; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1.0rem; color: #555; margin-bottom: 1rem; }
    .small-muted { color: #777; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧲 磁性测量数据处理平台（RunCenter）</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">选择任务 → 上传文件 → 配置参数 → 一键运行</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load():
    loaded = load_steps()
    by_cat: dict[str, list] = {}
    for item in loaded:
        by_cat.setdefault(item.step.meta.category, []).append(item)
    return by_cat


by_cat = _load()
if not by_cat:
    st.error("没有发现任何 steps。请检查 `magrun/steps/` 下是否存在 `step = ...`。")
    st.stop()

cats = list(by_cat.keys())
cat = st.selectbox("📌 选择任务类别", cats)
steps = by_cat[cat]
step_names = [s.step.meta.name for s in steps]
sel_name = st.selectbox("📋 选择具体功能", step_names)
sel = next(x for x in steps if x.step.meta.name == sel_name).step

with st.expander("📖 功能说明", expanded=True):
    st.markdown(sel.meta.description)
    st.markdown(f'<div class="small-muted">Step ID: <code>{sel.meta.id}</code></div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📂 上传文件", type=sel.meta.file_types, accept_multiple_files=True)
files: list[tuple[str, bytes]] = []
if uploaded:
    for f in uploaded:
        files.append((f.name, f.getvalue()))

st.markdown("### ⚙️ 参数设置")
params: dict[str, object] = {}
cols = st.columns(2)
for i, p in enumerate(sel.meta.params):
    with cols[i % 2]:
        if p.kind == "float":
            params[p.key] = st.number_input(p.label, value=float(p.default), format="%g", help=p.help)
        elif p.kind == "int":
            params[p.key] = int(st.number_input(p.label, value=int(p.default), step=1, help=p.help))
        elif p.kind == "bool":
            params[p.key] = st.checkbox(p.label, value=bool(p.default), help=p.help)
        elif p.kind == "select":
            opts = p.options or []
            idx = opts.index(p.default) if (p.default in opts) else 0
            params[p.key] = st.selectbox(p.label, opts, index=idx, help=p.help)
        else:
            params[p.key] = st.text_input(p.label, value=str(p.default), help=p.help)


if st.button("▶️ 开始处理", use_container_width=True, type="primary"):
    if not files:
        st.error("请先上传至少一个文件。")
        st.stop()

    with st.spinner("正在运行..."):
        outputs = sel.run(files=files, params=params)

    for note in outputs.notes:
        st.info(note)

    if outputs.tables:
        st.markdown("### 📊 结果预览")
        for name, df in outputs.tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True, height=360)

    if outputs.images:
        st.markdown("### 🖼️ 图片")
        for name, (fname, payload, mime) in outputs.images.items():
            st.markdown(f"**{name}**")
            st.image(payload)
            st.download_button(f"⬇️ 下载 {fname}", data=payload, file_name=fname, mime=mime)

    if outputs.downloads:
        st.markdown("### ⬇️ 下载文件")
        for name, (fname, payload, mime) in outputs.downloads.items():
            st.download_button(
                f"⬇️ {name}：{fname}",
                data=payload,
                file_name=fname,
                mime=mime,
                use_container_width=True,
            )

st.markdown("---")
st.caption("magnetic-run-center-web | 插件化 steps + Streamlit RunCenter")

