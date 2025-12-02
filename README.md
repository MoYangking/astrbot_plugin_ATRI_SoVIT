# astrbot_plugin_ATRI_SoVIT

ATRI SoVIT Gradio 接口插件，为 AstrBot 提供文本转语音(TTS)能力。

## 功能
- `/说 <文本>`：直接合成语音发送。
- 自动触发：按概率把 Bot 的回复转成语音（可配置概率和长度上限）。
- 接入 HuggingFace Gradio Space（默认：https://Moyang123-ATRI-test.hf.space），无本地 GPT-SoVITS 部署依赖。

## 安装
```bash
pip install -r requirements.txt
```

## 配置
- `gradio.space_url`：Gradio Space 地址，默认指向 ATRI 演示空间。
- `auto_config.send_record_probability` / `max_resp_text_len`：自动转语音的概率与文本长度限制。
- `tts_params.*`：对应 Gradio `predict` 的参数，除文本外都可在此修改：
  - `main_audio_label` / `prompt_text` / `prompt_language`
  - `text_language` / `how_to_cut`
  - `top_k` / `top_p` / `temperature`
  - `ref_free` / `speed` / `if_freeze`
  - `aux_audio_labels`（列表） / `sample_steps` / `if_sr` / `pause_second`

## 使用
- 自动模式：满足概率与长度限制时，Bot 回复会被替换成语音。
- 手动模式：`/说 你好`。

插件会把生成的语音复制到 `data/plugins_data/astrbot_plugin_ATRI_SoVIT/`，并删除 Gradio 临时目录。
