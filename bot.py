#!/usr/bin/env python3
import asyncio
import base64
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    "choices—don’t make assumptions. At the end, list action items only if relevant. "
    "Use exactly this heading for them: Cose da fare:. Put each action item on its own "
    "line starting with '- '. Do not use Markdown, bold, asterisks, HTML, or extra "
    "labels. If there are no action items, do not mention action items at all."
)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
BATCH_WINDOW_SECONDS = 0.8
DEFAULT_DISPLAY_TIMEZONE = "Europe/Rome"

FORMAT_ERROR_HINTS = (
    "unsupported audio format",
    "unsupported audio",
    "codec",
    "container",
    "format",
    "ogg",
    "opus",
)

TEMPORARY_MODEL_ERROR_HINTS = (
    "high demand",
    "try again later",
    "temporarily unavailable",
    "resource exhausted",
    "rate limit",
    "quota",
    "overloaded",
    "503",
    "429",
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

# Global queue for processing audio messages
audio_queue: Optional[asyncio.Queue] = None
audio_worker_task: Optional[asyncio.Task] = None


@dataclass
class AudioMessageTask:
    """Represents an audio message to be processed."""
    update: Update
    context: ContextTypes.DEFAULT_TYPE
    message_id: int
    chat_id: int
    file_id: str
    mime_type: str
    extension: str
    caption: str
    file_size: int
    file_name: Optional[str]
    telegram_timestamp: float
    whatsapp_timestamp: Optional[float]
    sender_label: Optional[str]
    source_timestamp: float


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
    """Enqueue audio messages for processing."""
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
        file_name = None
        default_extension = ".ogg"
    elif message.audio:
        source = message.audio
        mime_type = source.mime_type or "audio/mpeg"
        file_id = source.file_id
        file_size = source.file_size or 0
        unique_id = source.file_unique_id
        file_name = source.file_name
        default_extension = _extension_from_mime(mime_type) or ".mp3"
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("audio/"):
        source = message.document
        mime_type = source.mime_type or "application/octet-stream"
        file_id = source.file_id
        file_size = source.file_size or 0
        unique_id = source.file_unique_id
        file_name = source.file_name
        default_extension = _extension_from_mime(mime_type) or _extension_from_filename(source.file_name) or ".audio"
    else:
        await message.reply_text("Send an audio message and I'll transcribe it.")
        return

    logger.info(
        "Update %s: received audio message (chat_id=%s, mime=%s, size=%d bytes).",
        message.message_id,
        chat.id,
        mime_type,
        file_size,
    )

    logger.debug(
        "Update %s: received file_id=%s unique_id=%s mime=%s size=%s caption=%s",
        message.message_id,
        file_id,
        unique_id,
        mime_type,
        file_size,
        caption or "None",
    )
    if file_name:
        logger.debug("Update %s: source file name=%s", message.message_id, file_name)

    telegram_timestamp = message.date.replace(tzinfo=timezone.utc).timestamp() if message.date else time.time()
    display_tz = get_display_timezone()
    whatsapp_timestamp = parse_whatsapp_audio_timestamp(file_name, display_tz)
    if whatsapp_timestamp is not None:
        logger.debug(
            "Update %s: parsed WhatsApp timestamp %.0f from %s",
            message.message_id,
            whatsapp_timestamp,
            file_name,
        )

    sender_label = forwarded_sender_label(message) or caption or None
    source_timestamp = forwarded_message_timestamp(message) or whatsapp_timestamp or telegram_timestamp

    # Create task and add to queue
    task = AudioMessageTask(
        update=update,
        context=context,
        message_id=message.message_id,
        chat_id=chat.id,
        file_id=file_id,
        mime_type=mime_type,
        extension=default_extension,
        caption=caption,
        file_size=file_size,
        file_name=file_name,
        telegram_timestamp=telegram_timestamp,
        whatsapp_timestamp=whatsapp_timestamp,
        sender_label=sender_label,
        source_timestamp=source_timestamp,
    )

    if audio_queue is None:
        logger.error("Audio queue not initialized. Cannot process message.")
        await message.reply_text("Bot is not ready. Please try again.")
        return

    await audio_queue.put(task)
    logger.info(
        "Update %s: queued for processing (queue size: %d).",
        message.message_id,
        audio_queue.qsize(),
    )


async def process_audio_queue() -> None:
    """Worker coroutine that processes audio messages from the queue sequentially."""
    if audio_queue is None:
        logger.error("Audio queue not initialized.")
        return

    logger.info("Audio queue worker started.")
    while True:
        try:
            # Wait for a task from the queue
            first_task = await audio_queue.get()
            batch = [first_task]
            batch.extend(await collect_audio_batch())
            ordered_batch = sorted(batch, key=task_order_key)

            if len(ordered_batch) > 1:
                logger.info(
                    "Processing batch of %d audio messages ordered by timestamp.",
                    len(ordered_batch),
                )

            for task in ordered_batch:
                logger.info(
                    "Update %s: processing (queue size: %d).",
                    task.message_id,
                    audio_queue.qsize(),
                )

                message = task.update.effective_message
                if not message:
                    logger.warning("Update %s: message not found, skipping.", task.message_id)
                    audio_queue.task_done()
                    continue

                stop_event = asyncio.Event()
                typing_task = asyncio.create_task(typing_indicator(task.context, task.chat_id, stop_event))

                load_start = time.perf_counter()
                try:
                    await _process_audio_message(
                        context=task.context,
                        message=message,
                        file_id=task.file_id,
                        mime_type=task.mime_type,
                        extension=task.extension,
                        caption=task.caption,
                        file_size=task.file_size,
                        sender_label=task.sender_label,
                        source_timestamp=task.source_timestamp,
                    )
                except Exception as exc:  # noqa: BLE001
                    hint = remediation_hint(str(exc))
                    error_message = f"❌ Errore durante la trascrizione:\n\n{str(exc)}"
                    if hint and hint != "Verify network connectivity and environment configuration.":
                        error_message += f"\n\n💡 Suggerimento: {hint}"
                    logger.error(
                        "Processing failed for file_id=%s. Root cause: %s | Hint: %s",
                        task.file_id,
                        exc,
                        hint,
                        exc_info=True,
                    )
                    try:
                        await message.reply_text(error_message)
                    except Exception as send_exc:  # noqa: BLE001
                        logger.error("Failed to send error message: %s", send_exc, exc_info=True)
                finally:
                    stop_event.set()
                    await typing_task
                    load_duration = time.perf_counter() - load_start
                    logger.info(
                        "Update %s: processing completed (elapsed %.2fs, queue size: %d).",
                        task.message_id,
                        load_duration,
                        audio_queue.qsize(),
                    )
                    logger.debug("Update %s: total handler duration %.2fs", task.message_id, load_duration)
                    audio_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Audio queue worker cancelled.")
            break
        except Exception as exc:  # noqa: BLE001
            if "is bound to a different event loop" in str(exc):
                logger.warning("Stopping audio queue worker due to event loop mismatch during shutdown.")
                break
            logger.error("Error in audio queue worker: %s", exc, exc_info=True)


async def collect_audio_batch() -> list[AudioMessageTask]:
    """Collects additional audio tasks that arrive in a short burst."""
    if audio_queue is None:
        return []

    tasks: list[AudioMessageTask] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BATCH_WINDOW_SECONDS

    while True:
        timeout = deadline - loop.time()
        if timeout <= 0:
            break
        try:
            # Extend the window each time a new item arrives so grouped forwards
            # can be reordered together.
            task = await asyncio.wait_for(audio_queue.get(), timeout=timeout)
            tasks.append(task)
            deadline = loop.time() + BATCH_WINDOW_SECONDS
        except asyncio.TimeoutError:
            break

    return tasks


def parse_whatsapp_audio_timestamp(file_name: Optional[str], display_tz: ZoneInfo) -> Optional[float]:
    """
    Parse WhatsApp audio naming convention, e.g. AUDIO-2026-02-19-19-27-38.m4a.
    Returns an epoch timestamp for ordering if pattern is present.
    """
    if not file_name:
        return None

    match = re.search(r"(?:AUDIO|PTT)-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", file_name, re.IGNORECASE)
    if not match:
        return None

    try:
        year, month, day, hour, minute, second = (int(part) for part in match.groups())
        return datetime(year, month, day, hour, minute, second, tzinfo=display_tz).timestamp()
    except ValueError:
        return None


def get_display_timezone() -> ZoneInfo:
    timezone_name = os.getenv("DISPLAY_TIMEZONE", DEFAULT_DISPLAY_TIMEZONE)
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Invalid DISPLAY_TIMEZONE=%s; falling back to %s.", timezone_name, DEFAULT_DISPLAY_TIMEZONE)
        return ZoneInfo(DEFAULT_DISPLAY_TIMEZONE)


def forwarded_sender_label(message) -> Optional[str]:
    origin = getattr(message, "forward_origin", None)
    if not origin:
        return None

    sender_user = getattr(origin, "sender_user", None)
    if sender_user:
        return sender_user.full_name

    sender_user_name = getattr(origin, "sender_user_name", None)
    if sender_user_name:
        return sender_user_name

    sender_chat = getattr(origin, "sender_chat", None) or getattr(origin, "chat", None)
    if sender_chat:
        return getattr(sender_chat, "title", None) or getattr(sender_chat, "full_name", None)

    author_signature = getattr(origin, "author_signature", None)
    if author_signature:
        return author_signature

    return None


def forwarded_message_timestamp(message) -> Optional[float]:
    origin = getattr(message, "forward_origin", None)
    origin_date = getattr(origin, "date", None) if origin else None
    if not origin_date:
        return None
    if origin_date.tzinfo is None:
        origin_date = origin_date.replace(tzinfo=timezone.utc)
    return origin_date.timestamp()


def build_telegram_response(text: str, sender_label: Optional[str], source_timestamp: float) -> str:
    transcript = format_transcription((text or "").strip())
    if not transcript:
        return ""

    footer = format_source_footer(sender_label, source_timestamp)
    return f"{transcript}\n\n----\n{footer}" if footer else transcript


def format_transcription(text: str) -> str:
    lines = text.splitlines()
    formatted_lines: list[str] = []
    in_action_items = False

    for line in lines:
        stripped = line.strip()
        action_heading = re.fullmatch(
            r"(?:\*\*)?\s*(?:Action Items|Cose da fare):?\s*(?:\*\*)?",
            stripped,
            re.IGNORECASE,
        )
        if action_heading:
            formatted_lines.append("Cose da fare:")
            in_action_items = True
            continue

        bullet_match = re.match(r"^\s*(?:[-*•]\s+)(.+)$", line)
        if in_action_items and bullet_match:
            formatted_lines.append(f"• {bullet_match.group(1).strip()}")
            continue

        formatted_lines.append(line)

    return "\n".join(formatted_lines).strip()


def format_source_footer(sender_label: Optional[str], source_timestamp: float) -> str:
    formatted_date = format_italian_datetime(source_timestamp)
    if sender_label:
        return f"Da {sender_label}, {formatted_date}"
    return f"Ricevuto {formatted_date}"


def format_italian_datetime(timestamp: float) -> str:
    display_tz = get_display_timezone()
    local_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(display_tz)
    weekdays = (
        "lunedì",
        "martedì",
        "mercoledì",
        "giovedì",
        "venerdì",
        "sabato",
        "domenica",
    )
    months = (
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    )
    weekday = weekdays[local_dt.weekday()]
    month = months[local_dt.month - 1]
    return f"{weekday} {local_dt.day} {month} {local_dt.year} alle {local_dt:%H.%M}"


def task_order_key(task: AudioMessageTask) -> tuple[float, float, int]:
    """
    Order by WhatsApp timestamp when available, otherwise Telegram receive time.
    """
    primary_timestamp = task.whatsapp_timestamp if task.whatsapp_timestamp is not None else task.telegram_timestamp
    return (primary_timestamp, task.telegram_timestamp, task.message_id)


async def _process_audio_message(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    file_id: str,
    mime_type: str,
    extension: str,
    caption: str,
    file_size: int,
    sender_label: Optional[str],
    source_timestamp: float,
) -> None:
    bot = context.bot
    tg_file = await bot.get_file(file_id)

    prompt = DEFAULT_PROMPT

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

    result_text = build_telegram_response(result_text, sender_label, source_timestamp)
    if not result_text:
        logger.info("Update %s: no speech detected in audio.", message.message_id)
        await message.reply_text("❌ Non è stato rilevato alcun discorso nell'audio. Riprova con un altro messaggio vocale.")
        return

    logger.info(
        "Update %s: transcription completed (%d characters).",
        message.message_id,
        len(result_text),
    )

    # Send the transcription as plain text.
    if len(result_text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        # Split into chunks to stay within Telegram limits.
        chunks = _chunk_text(result_text, TELEGRAM_MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            await message.reply_text(chunk)
    else:
        await message.reply_text(result_text)


def transcribe_audio_with_gemini(source_path: str, mime_type: str, prompt: str) -> str:
    api_key = require_env("AI_MODEL_API_KEY")
    model = require_env("AI_MODEL")
    fallback_model = os.getenv("AI_MODEL_FALLBACK", "").strip()
    active_model = model

    attempt = 1
    logger.debug(
        "Preparing Gemini request attempt %d: model=%s mime=%s path=%s prompt_chars=%d",
        attempt,
        model,
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

    if result.error and fallback_model and fallback_model != model and should_retry_with_fallback_model(result.error):
        attempt += 1
        logger.info("Primary model %s unavailable; retrying with fallback model %s.", model, fallback_model)
        active_model = fallback_model
        result = call_gemini(api_key, fallback_model, source_path, mime_type, prompt)
        logger.debug(
            "Gemini attempt %d (fallback): status=%s duration=%.2fs error=%s",
            attempt,
            result.status_code,
            result.duration or 0.0,
            result.error or "None",
        )
        if result.ok and result.text:
            logger.debug(
                "Gemini transcription succeeded with fallback model %s (%d chars).",
                fallback_model,
                len(result.text),
            )
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
            result = call_gemini(api_key, active_model, converted_path, "audio/wav", prompt)
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


def should_retry_with_fallback_model(error_message: str) -> bool:
    lowered = error_message.lower()
    return any(keyword in lowered for keyword in TEMPORARY_MODEL_ERROR_HINTS)


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
    chunks: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(line) > chunk_size:
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.extend(line[i : i + chunk_size].rstrip() for i in range(0, len(line), chunk_size))
            continue

        if len(current) + len(line) > chunk_size:
            chunks.append(current.rstrip())
            current = line
        else:
            current += line

    if current:
        chunks.append(current.rstrip())

    return chunks


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
    global audio_queue, audio_worker_task

    load_dotenv()
    debug_enabled = parse_bool(os.getenv("DEBUG"), default=False)
    configure_logging(debug_enabled)

    # Allow using a test bot token for local development
    # Set TELEGRAM_BOT_TOKEN_TEST in your local .env to use a different bot
    test_token = os.getenv("TELEGRAM_BOT_TOKEN_TEST")
    if test_token:
        telegram_token = test_token
        logger.info("Using TEST bot token for local development.")
    else:
        telegram_token = require_env("TELEGRAM_BOT_TOKEN")
        logger.info("Using PRODUCTION bot token.")

    # Start the queue worker when the application starts
    async def post_init(application: Application) -> None:
        """Start the queue worker after the application is initialized."""
        global audio_queue, audio_worker_task
        audio_queue = asyncio.Queue()
        audio_worker_task = asyncio.create_task(process_audio_queue())

    async def post_shutdown(application: Application) -> None:
        """Stop worker gracefully when the application shuts down."""
        global audio_worker_task
        if audio_worker_task is not None:
            audio_worker_task.cancel()
            try:
                await audio_worker_task
            except asyncio.CancelledError:
                pass
            audio_worker_task = None

    application = (
        Application.builder()
        .token(telegram_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    audio_handler = MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio,
    )
    application.add_handler(audio_handler)

    logger.info("Bot started. Waiting for audio messages.")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
