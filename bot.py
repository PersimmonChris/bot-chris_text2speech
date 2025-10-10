#!/usr/bin/env python3
import asyncio
import base64
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

DEFAULT_PROMPT = (
    "Convert this to text while keeping the tone of voice. Remove only the ‘ehm’ and "
    "other filler words—if in doubt, keep them. Preserve the speaker’s original word "
    "choices—don’t make assumptions. At the end, list any action items if relevant "
    "(for example, if the speaker mentioned something that needs fixing)."
    "if no action items, don't mention NO ACTION ITEMS."
)

PROCESSING_MESSAGE = "Processing audio..."
FAILURE_MESSAGE = "Couldn't process that audio. Tap to retry."
NO_SPEECH_MESSAGE = "I couldn't hear any speech in that recording. Try sending it again?"
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

FORMAT_ERROR_HINTS = (
    "unsupported audio format",
    "unsupported audio",
    "codec",
    "container",
    "format",
    "ogg",
    "opus",
)


@dataclass
class GeminiResult:
    ok: bool
    text: Optional[str] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    duration: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


logger = logging.getLogger("chris_text2speech")


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(debug_enabled: bool) -> None:
    level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Reduce noise from dependency loggers unless debug is explicitly requested.
    if not debug_enabled:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("telegram").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def typing_indicator(context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event) -> None:
    """Continuously sends typing action while work is in progress."""
    try:
        while not stop_event.is_set():
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Typing indicator failed: %s", exc)
                break
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue
    finally:
        stop_event.set()


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    chat = update.effective_chat
    if not chat:
        return

    caption = (message.caption or "").strip()

    if message.voice:
        source = message.voice
        mime_type = source.mime_type or "audio/ogg"
        file_id = source.file_id
        file_size = source.file_size or 0
        unique_id = source.file_unique_id
        default_extension = ".ogg"
    elif message.audio:
        source = message.audio
        mime_type = source.mime_type or "audio/mpeg"
        file_id = source.file_id
        file_size = source.file_size or 0
        unique_id = source.file_unique_id
        default_extension = _extension_from_mime(mime_type) or ".mp3"
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("audio/"):
        source = message.document
        mime_type = source.mime_type or "application/octet-stream"
        file_id = source.file_id
        file_size = source.file_size or 0
        unique_id = source.file_unique_id
        default_extension = _extension_from_mime(mime_type) or _extension_from_filename(source.file_name) or ".audio"
    else:
        await message.reply_text("Send an audio message and I'll transcribe it.")
        return

    processing_message = await message.reply_text(PROCESSING_MESSAGE)

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(typing_indicator(context, chat.id, stop_event))

    logger.debug(
        "Update %s: received file_id=%s unique_id=%s mime=%s size=%s caption=%s",
        message.message_id,
        file_id,
        unique_id,
        mime_type,
        file_size,
        caption or "None",
    )

    load_start = time.perf_counter()
    try:
        await _process_audio_message(
            context=context,
            message=message,
            processing_message=processing_message,
            file_id=file_id,
            mime_type=mime_type,
            extension=default_extension,
            caption=caption,
            file_size=file_size,
        )
    except Exception as exc:  # noqa: BLE001
        hint = remediation_hint(str(exc))
        logger.error(
            "Processing failed for file_id=%s. Root cause: %s | Hint: %s",
            file_id,
            exc,
            hint,
            exc_info=True,
        )
        await processing_message.edit_text(FAILURE_MESSAGE)
    finally:
        stop_event.set()
        await typing_task
        load_duration = time.perf_counter() - load_start
        logger.debug("Update %s: total handler duration %.2fs", message.message_id, load_duration)


async def _process_audio_message(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    processing_message,
    file_id: str,
    mime_type: str,
    extension: str,
    caption: str,
    file_size: int,
) -> None:
    bot = context.bot
    tg_file = await bot.get_file(file_id)

    prompt = DEFAULT_PROMPT + ("\n\n" + caption if caption else "")

    with tempfile.TemporaryDirectory(prefix="tts_") as tmp_dir:
        original_path = os.path.join(tmp_dir, f"source{extension}")

        download_start = time.perf_counter()
        await tg_file.download_to_drive(custom_path=original_path)
        download_duration = time.perf_counter() - download_start
        logger.debug(
            "Downloaded to %s (%d bytes) in %.2fs",
            original_path,
            file_size,
            download_duration,
        )

        result_text = await asyncio.to_thread(
            transcribe_audio_with_gemini,
            original_path,
            mime_type,
            prompt,
        )

    result_text = (result_text or "").strip()
    if not result_text:
        await processing_message.edit_text(NO_SPEECH_MESSAGE)
        return

    if len(result_text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        # Split into chunks to stay within Telegram limits.
        chunks = _chunk_text(result_text, TELEGRAM_MAX_MESSAGE_LENGTH)
        await processing_message.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await message.reply_text(chunk)
    else:
        await processing_message.edit_text(result_text)


def transcribe_audio_with_gemini(source_path: str, mime_type: str, prompt: str) -> str:
    api_key = require_env("AI_MODEL_API_KEY")
    model = require_env("AI_MODEL")

    attempt = 1
    logger.debug(
        "Preparing Gemini request attempt %d: mime=%s path=%s prompt_chars=%d",
        attempt,
        mime_type,
        source_path,
        len(prompt),
    )
    result = call_gemini(api_key, model, source_path, mime_type, prompt)
    logger.debug(
        "Gemini attempt %d: status=%s duration=%.2fs error=%s",
        attempt,
        result.status_code,
        result.duration or 0.0,
        result.error or "None",
    )

    if result.ok and result.text:
        logger.debug("Gemini transcription succeeded on attempt %d (%d chars).", attempt, len(result.text))
        return result.text

    if result.error and should_retry_with_wav(result.error):
        attempt += 1
        converted_path, convert_duration, ffmpeg_cmd = convert_to_wav(source_path)
        try:
            logger.debug(
                "Conversion via ffmpeg (%.2fs): %s",
                convert_duration,
                " ".join(ffmpeg_cmd),
            )
            result = call_gemini(api_key, model, converted_path, "audio/wav", prompt)
            logger.debug(
                "Gemini attempt %d (WAV): status=%s duration=%.2fs error=%s",
                attempt,
                result.status_code,
                result.duration or 0.0,
                result.error or "None",
            )
            if result.ok and result.text:
                logger.debug(
                    "Gemini transcription succeeded after conversion (%d chars).",
                    len(result.text),
                )
                return result.text
        finally:
            try:
                os.remove(converted_path)
            except OSError:
                logger.debug("Failed to remove temporary WAV file at %s", converted_path)

    root_cause = result.error or "Unknown error from Gemini"
    raise RuntimeError(root_cause)


def call_gemini(api_key: str, model: str, audio_path: str, mime_type: str, prompt: str) -> GeminiResult:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ]
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(
            url,
            params={"key": api_key},
            headers=headers,
            json=payload,
            timeout=120,
        )
        duration = time.perf_counter() - start_time
    except requests.RequestException as exc:
        return GeminiResult(ok=False, error=str(exc), duration=None)

    status_code = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = None

    if status_code == 200 and data:
        candidate = _extract_candidate(data)
        if candidate:
            return GeminiResult(
                ok=True,
                text=candidate,
                raw=data,
                status_code=status_code,
                duration=duration,
            )
        error_message = "Gemini response missing transcript text."
        return GeminiResult(
            ok=False,
            error=error_message,
            raw=data,
            status_code=status_code,
            duration=duration,
        )

    if data and "error" in data:
        error_obj = data["error"]
        message = error_obj.get("message", "Gemini API returned an error.")
        return GeminiResult(
            ok=False,
            error=message,
            raw=data,
            status_code=status_code,
            duration=duration,
        )

    return GeminiResult(
        ok=False,
        error=f"Gemini API error {status_code}: {response.text}",
        status_code=status_code,
        duration=duration,
    )


def _extract_candidate(data: Dict[str, Any]) -> Optional[str]:
    candidates = data.get("candidates") or []
    if not candidates:
        prompt_feedback = data.get("promptFeedback", {})
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            return None
        return None

    first = candidates[0]
    content = first.get("content", {})
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if "text" in part]
    text_combined = "\n".join(piece for piece in texts if piece)
    return text_combined if text_combined.strip() else None


def should_retry_with_wav(error_message: str) -> bool:
    lowered = error_message.lower()
    return any(keyword in lowered for keyword in FORMAT_ERROR_HINTS)


def convert_to_wav(source_path: str) -> tuple[str, float, list[str]]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for format conversion but was not found in PATH.")

    converted_fd, converted_path = tempfile.mkstemp(suffix=".wav", prefix="tts_converted_")
    os.close(converted_fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        converted_path,
    ]
    start_time = time.perf_counter()
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    duration = time.perf_counter() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg conversion failed (exit {process.returncode}): {process.stderr.strip()}"
        )

    return converted_path, duration, cmd


def _extension_from_mime(mime_type: Optional[str]) -> Optional[str]:
    if not mime_type:
        return None
    ext = mimetypes.guess_extension(mime_type)
    if ext == ".oga":
        # Telegram voice notes often map to .ogg despite .oga guess.
        return ".ogg"
    return ext


def _extension_from_filename(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    _, ext = os.path.splitext(filename)
    return ext


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def remediation_hint(error_message: str) -> str:
    lowered = error_message.lower()
    if "ffmpeg" in lowered and "not found" in lowered:
        return "Install FFmpeg and ensure it is available on PATH."
    if any(keyword in lowered for keyword in ("unsupported", "codec", "format", "container")):
        return "Gemini rejected the audio format; try converting to 16 kHz mono WAV."
    if "permission" in lowered:
        return "Check filesystem permissions for the working directory."
    if "timeout" in lowered or "deadline" in lowered:
        return "Request timed out; check network connectivity or retry with a shorter clip."
    if "unauthorized" in lowered or "permission_denied" in lowered:
        return "Verify the Gemini API key and ensure the chosen model is enabled."
    return "Verify network connectivity and environment configuration."


def main() -> None:
    load_dotenv()
    debug_enabled = parse_bool(os.getenv("DEBUG"), default=False)
    configure_logging(debug_enabled)

    telegram_token = require_env("TELEGRAM_BOT_TOKEN")

    application = Application.builder().token(telegram_token).build()

    audio_handler = MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio,
    )
    application.add_handler(audio_handler)

    logger.info("Bot started. Waiting for audio messages.")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
