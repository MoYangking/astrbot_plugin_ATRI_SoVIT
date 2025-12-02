import asyncio
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from gradio_client import Client

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Record
import astrbot.core.message.components as Comp
from astrbot.core.platform import AstrMessageEvent

# 语音文件保存目录
SAVED_AUDIO_DIR = Path("./data/plugins_data/astrbot_plugin_ATRI_SoVIT")
SAVED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@register(
    "astrbot_plugin_ATRI_SoVIT",
    "MoYang",
    "ATRI_SoVIT对接插件",
    "1.1.0",
    "https://github.com/MoYangking/astrbot_plugin_ATRI_SoVIT",
)
class GPTSoVITSPlugin(Star):
    """
    ATRI SoVIT Gradio 接口插件：
    - 自动按概率将 bot 回复转成语音
    - 指令 /说 直接合成语音
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)

        gradio_config: Dict[str, Any] = config.get("gradio", {})
        self.space_url: str = gradio_config.get(
            "space_url", "https://Moyang123-ATRI-test.hf.space"
        )
        self.client = Client(self.space_url)

        auto_config: Dict[str, Any] = config.get("auto_config", {})
        self.send_record_probability: float = auto_config.get(
            "send_record_probability", 0.15
        )
        self.max_resp_text_len: int = auto_config.get("max_resp_text_len", 50)

        tts_config: Dict[str, Any] = config.get("tts_params", {})
        self.tts_params: Dict[str, Any] = {
            "main_audio_label": tts_config.get(
                "main_audio_label",
                "ATR_b401_088.wav | Japanese | それは…………おそらく本当です。思い出せるメモリー...",
            ),
            "prompt_text": tts_config.get(
                "prompt_text",
                "それは…………おそらく本当です。思い出せるメモリーの断片に、うっすらと叱られた記憶が残っています",
            ),
            "prompt_language": tts_config.get("prompt_language", "Japanese"),
            "text_language": tts_config.get("text_language", "English"),
            "how_to_cut": tts_config.get(
                "how_to_cut", "Slice once every 4 sentences"
            ),
            "top_k": tts_config.get("top_k", 15),
            "top_p": tts_config.get("top_p", 1),
            "temperature": tts_config.get("temperature", 1),
            "ref_free": tts_config.get("ref_free", False),
            "speed": tts_config.get("speed", 1),
            "if_freeze": tts_config.get("if_freeze", False),
            "aux_audio_labels": tts_config.get(
                "aux_audio_labels",
                [
                    "ATR_b401_096.wav | Japanese | 甘えんぼアトリになって眠ってもよろしいでしょうか",
                    "ATR_b401_059.wav | Japanese | はい。有用性が獲得できますから",
                ],
            ),
            "sample_steps": tts_config.get("sample_steps", 8),
            "if_sr": tts_config.get("if_sr", False),
            "pause_second": tts_config.get("pause_second", 0.3),
        }

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 概率控制：随机数大于阈值则不处理，直接返回
        if random.random() > self.send_record_probability:
            return

        chain = event.get_result().chain
        seg = chain[0]

        if not (len(chain) == 1 and isinstance(seg, Comp.Plain)):
            return

        resp_text = seg.text

        if len(resp_text) > self.max_resp_text_len:
            return

        save_path = await self.tts_inference(event=event, text=resp_text)

        if save_path is None:
            logger.error("TTS任务执行失败！")
            return

        chain.clear()
        chain.append(Record.fromFileSystem(save_path))

    @filter.command("说")
    async def on_command(
        self,
        event: AstrMessageEvent,
        send_text: str | int | None = None,
    ):
        if not send_text:
            yield event.plain_result("未提供文本")
            return

        text = str(send_text)
        save_path = await self.tts_inference(event=event, text=text)

        if save_path is None:
            logger.error("TTS任务执行失败！")
            return

        chain = [Record.fromFileSystem(save_path)]
        yield event.chain_result(chain)  # type: ignore

    def generate_file_name(
        self, event: Optional[AstrMessageEvent], text: str, extension: str
    ) -> str:
        group_id = event.get_group_id() if event else "0"
        sender_id = event.get_sender_id() if event else "0"

        sanitized_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s]", "", text)
        limit_text = sanitized_text.strip()[:30] or "audio"

        ext = extension if extension.startswith(".") else f".{extension}"
        file_name = f"{group_id}_{sender_id}_{limit_text}{ext}"
        return file_name

    async def tts_inference(self, event: AstrMessageEvent, text: str) -> str | None:
        result_file = await self._predict_audio_path(text)
        if result_file is None:
            return None

        extension = result_file.suffix or ".wav"
        file_name = self.generate_file_name(event, text, extension)
        target_path = (SAVED_AUDIO_DIR / file_name).resolve()

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_file, target_path)
        except Exception as e:
            logger.error(f"保存 TTS 文件失败：{e}")
            return None

        tmp_dir = result_file.parent
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return str(target_path)

    async def tts_sever(self, text: str, file_name: str) -> str | None:
        result_file = await self._predict_audio_path(text)
        if result_file is None:
            return None

        extension = result_file.suffix or ".wav"
        name = file_name
        if not Path(file_name).suffix:
            name = f"{file_name}{extension}"

        target_path = (SAVED_AUDIO_DIR / name).resolve()

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_file, target_path)
        except Exception as e:
            logger.error(f"保存 TTS 文件失败：{e}")
            return None

        tmp_dir = result_file.parent
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return str(target_path)

    async def _predict_audio_path(self, text: str) -> Optional[Path]:
        aux_audio_labels = self.tts_params.get("aux_audio_labels", [])
        if isinstance(aux_audio_labels, str):
            aux_audio_labels = [aux_audio_labels] if aux_audio_labels else []
        if not isinstance(aux_audio_labels, list):
            aux_audio_labels = list(aux_audio_labels)

        try:
            result_path = await asyncio.to_thread(
                self.client.predict,
                self.tts_params["main_audio_label"],
                self.tts_params["prompt_text"],
                self.tts_params["prompt_language"],
                text,
                self.tts_params["text_language"],
                self.tts_params["how_to_cut"],
                self.tts_params["top_k"],
                self.tts_params["top_p"],
                self.tts_params["temperature"],
                bool(self.tts_params["ref_free"]),
                self.tts_params["speed"],
                bool(self.tts_params["if_freeze"]),
                aux_audio_labels,
                self.tts_params["sample_steps"],
                bool(self.tts_params["if_sr"]),
                self.tts_params["pause_second"],
                api_name="/get_tts_wav",
            )
        except Exception as e:
            logger.error(f"TTS 调用失败：{e}")
            return None

        if not result_path:
            logger.error("TTS 未返回结果")
            return None

        return Path(str(result_path))
