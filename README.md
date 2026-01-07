# 离线实时语音翻译（Web）

本项目提供一个本地 Web UI（FastAPI），用于**离线运行**的实时语音识别与翻译：

- 浏览器麦克风采集（本机 `localhost` 可直接 `getUserMedia`）
- `webrtcvad` 做低延迟 VAD 分段（连续静音超过阈值切段）
- 离线 ASR：`faster-whisper`（支持本地模型目录，如 `faster-whisper-large-v3-turbo` / `faster-whisper-small`）
- 通过本地 OpenAI 兼容接口调用 HY-MT 翻译模型：
  - `mic_in/file_in`：未知语种 A → 中文
  - `mic_out/text_out`：中文 → 语种 L
- 本机 TTS 播放：优先 SAPI（pywin32），否则 PowerShell/System.Speech

> 硬性约束：除 `http://localhost:1234/v1/chat/completions` 外不依赖任何外部云服务。

---

## 1. 环境要求

- Windows 10/11
- Python 3.11
- NVIDIA GPU（推荐：用于 ASR 加速；若 GPU 初始化/推理失败会自动降级 CPU）
- ASR 依赖（本地运行）：`faster-whisper`（GPU 需 CUDA/cuDNN，缺失会自动走 CPU）
- 本地翻译接口已就绪：
  - `POST http://localhost:1234/v1/chat/completions`
  - `model = "hy-mt1.5-1.8b"`
  - 返回 OpenAI 兼容 JSON（`choices[0].message.content`）

主要语言范围：中文、英语、日语、韩语、西班牙语、法语、德语、阿拉伯语；其它语言也允许但不保证效果。

---

## 2. 安装

建议使用虚拟环境：

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

说明：
- 本项目默认 ASR 使用 `config.json` 中的 `asr.model_path`（可填写本地模型目录或 `small` 等内置模型名）。
- `faster-whisper` 已写入 `requirements.txt`；若安装失败，请先确认本机环境可编译/安装其依赖（尤其是 Windows + CUDA/CPU 相关组件）。

### 2.1 模型下载（必须）

为减小仓库体积，`faster-whisper-*` 模型目录不会随仓库上传到 GitHub。请在本机自行下载并放到项目根目录（或任意目录），然后在 `config.json` 的 `asr.model_path` 指向该目录。

推荐模型（按体积由大到小）：
- `faster-whisper-large-v3-turbo`
- `faster-whisper-small`

获取方式（任选其一）：
- 从 Hugging Face 下载并解压到本地目录：
  - https://huggingface.co/Systran/faster-whisper-large-v3-turbo
  - https://huggingface.co/Systran/faster-whisper-small
- 或在可联网环境运行时，将 `asr.model_path` 设置为内置模型名（如 `small`），首次运行会自动下载并缓存；离线环境请使用本地模型目录。

离线运行注意：
- 建议使用本地模型目录（例如放在项目根目录：`faster-whisper-large-v3-turbo` / `faster-whisper-small`），然后在 `config.json` 里设置 `asr.model_path` 为该目录名。
- 若 `asr.model_path` 填的是内置模型名（如 `small`），需要能从网络下载模型；离线环境请改用本地模型目录。

---

## 3. 运行

```powershell
python -m translation_app
```

或直接：

```powershell
python translation_app/web/server.py
```

然后用浏览器打开 `http://127.0.0.1:8000/`。

界面左上角可在 **调试模式 / 简单模式** 间切换：
- 调试模式：保留“音频输入/文件输入/音频输出/文字输出”按钮与底部输入框。
- 简单模式：仅保留底部两个大按钮「听 / 说」，需要按住按钮才会录音；录音过程中会按间隔实时刷新 ASR 文本，松开即结束并输出最终结果。

---

## 4. 配置文件（config.json）

本项目不再通过命令行传参配置运行参数，统一使用项目根目录下的 `config.json`。

关键字段（对应 `config.json` 内）：

- `asr.backend`：`whisper`
- `asr.device`：`cuda` 或 `cpu`
- `asr.model_path`：模型目录或内置模型名（如 `faster-whisper-large-v3-turbo` / `faster-whisper-small` / `small`）
- `vad.aggressiveness` / `vad.silence_ms` / `vad.min_segment_ms`
- `ui.simple_partial_asr`：简单模式录音时的“实时 ASR”刷新（可关闭/调参）
- `language.prob_threshold`
- `translate.api_url`
- `reply_mode`：当前 Web UI 固定为对话式交互（输入中文后翻译到语种 L）
- `tts.enabled`：是否启用 TTS

`ui.simple_partial_asr` 子项：
- `enabled`：是否启用（默认 `true`）
- `interval_s`：两次刷新最小间隔（秒）
- `min_dur_s`：累计录音达到该时长后才开始刷新（秒）
- `min_step_s`：新增录音达到该时长后才触发下一次刷新（秒）
- `max_dur_s`：超过该时长不再做实时刷新（仍会在松开后做一次最终识别）

---

## 6. 常见问题排查

1) **翻译接口不可用**
- 现象：提示翻译失败
- 行为：程序仍继续做 ASR；翻译恢复后会自动继续翻译
- 排查：确认 `http://localhost:1234/v1/chat/completions` 正常返回 OpenAI 兼容 JSON，且 `model=hy-mt1.5-1.8b`

2) **GPU 不可用或推理失败**
- 行为：会自动降级 CPU，并在日志中提示
- Windows 常见原因：缺少 `cudnn_ops64_9.dll`（cuDNN 9）。解决：安装与当前 CUDA/驱动匹配的 cuDNN 9，并把其 `bin` 目录加入系统 `PATH`；或在 `config.json` 设置 `asr.device=cpu`

如果你已经在其它环境/软件目录里找到了该 DLL（例如某个环境的 `torch\lib` 目录），也可以用环境变量临时指定：

```powershell
# README.md
$env:CUDNN_DLL_DIR="C:\Users\Administrator\Documents\Code\PythonCode\test\.venv\Lib\site-packages\torch\lib"
python -m translation_app
```

3) **浏览器无法使用麦克风**
- 排查：Windows 设置 → 隐私与安全 → 麦克风（允许桌面应用访问）；浏览器站点权限允许麦克风；确认是 `http://127.0.0.1:8000/` 或 `http://localhost:8000/`
