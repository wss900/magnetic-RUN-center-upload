# magnetic-run-center-web

一个可扩展的磁性测量数据处理平台：**Streamlit RunCenter + 插件化 steps**。

## 运行

```powershell
cd magnetic-run-center-web
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run runcenter_app.py
```

## 插件化 steps（1 分钟接入）

- 把新功能写成一个 `step` 文件放进 `magrun/steps/`
- 在文件里暴露顶层变量：`step = YourStep()`
- RunCenter 启动时会自动扫描并出现在菜单里（无需改 `runcenter_app.py`）

目前已有 steps：

- **PPMS 角度扫描拟合（多段磁场）**
- **谐波斜率/曲率分析（First 二次拟合取曲率，Second 线性拟合取斜率）**


## 如何向 AI 描述一个新 step（推荐模板）

当你希望我（或其它 AI）为 RunCenter 新增一个功能时，请尽量按下面格式提供信息。这样生成的 step 能“一次成功接入”，并且无需修改 `runcenter_app.py`。

### 最推荐话术模板（复制后填空）

```text
请为我的 Streamlit RunCenter 新增一个 step（不要改 runcenter_app.py）：

- 功能名（菜单显示）：<name>
- 类别：<category>
- 文件类型：<txt/xlsx/zip>，<单文件/多文件>
- 输入格式示例（贴 5-10 行原始数据）：
<sample>
- 参数（key/类型/默认值/说明）：
<params>
- 处理逻辑（1-5 条）：
<logic>
- 输出：
  - 表：<列名/说明>
  - 文件：<zip/excel/png> 文件名
  - 失败/废弃规则：<QC>
- 接入要求：生成 `magrun/steps/<file>.py`，文件末尾必须有 `step = XxxStep()`
```

### 接入检查清单

- 新文件是否放在 `magrun/steps/`？
- 文件末尾是否有 `step = ...`？
- `import` step 时是否无副作用（不要在 step 里写 Streamlit UI 调用）？
- 本地运行 `streamlit run runcenter_app.py` 后，新功能是否出现在菜单中？

别感冒

