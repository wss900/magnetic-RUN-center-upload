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

