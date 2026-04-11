"""
voice/jarvis.py
────────────────────────────────────────────────────────────────
Standalone voice runtime for a Jarvis-style assistant.

Design goals:
- Keep Streamlit app untouched and stable.
- Add clap wake + continuous voice conversation + spoken responses.
- Reuse existing routing, guardrails, actions, and RAG chain logic.

Run:
    python -m voice.jarvis

Dependencies (install once):
    pip install sounddevice speechrecognition pyttsx3
"""

from __future__ import annotations

import re
import sys
import time
import queue
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import load_settings, load_prompts, setup_logging
from generation.rag_chain import RAGChain
from agent.intent_router import IntentRouter
from agent.session_state import SessionState
from agent.actions import (
    handle_create_ticket,
    handle_check_ticket,
    handle_check_billing,
    handle_register_user,
    handle_upgrade_plan,
    handle_close_tickets,
    handle_close_ticket_by_id,
    handle_list_accounts,
    is_ticket_flow_active,
    is_register_flow_active,
)
from agent.guardrails import get_guardrail_response, handle_insufficient_evidence


def _require_voice_dependencies():
    missing = []
    try:
        import numpy as np  # noqa: F401
    except Exception:
        missing.append("numpy")

    try:
        import sounddevice as sd  # noqa: F401
    except Exception:
        missing.append("sounddevice")

    try:
        import speech_recognition as sr  # noqa: F401
    except Exception:
        missing.append("speechrecognition")

    try:
        import pyttsx3  # noqa: F401
    except Exception:
        missing.append("pyttsx3")

    if missing:
        raise RuntimeError(
            "Missing voice dependencies: "
            + ", ".join(missing)
            + "\nInstall with: pip install sounddevice speechrecognition pyttsx3"
        )


@dataclass
class ClapConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    min_gap_sec: float = 0.10
    max_double_clap_sec: float = 1.20
    threshold_multiplier: float = 6.0
    abs_floor: float = 1500.0


class ClapDetector:
    def __init__(self, cfg: ClapConfig | None = None):
        _require_voice_dependencies()
        import numpy as np
        import sounddevice as sd

        self.np = np
        self.sd = sd
        self.cfg = cfg or ClapConfig()

    def wait_for_double_clap(self, timeout_sec: Optional[float] = None) -> bool:
        """
        Wait for a double clap using RMS spike detection.
        Returns True on detection, False on timeout.
        """
        cfg = self.cfg
        start = time.monotonic()

        last_spike = -999.0
        spikes: list[float] = []
        noise = 120.0

        with self.sd.RawInputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=cfg.block_size,
        ) as stream:
            while True:
                if timeout_sec is not None and (time.monotonic() - start) > timeout_sec:
                    return False

                data, overflowed = stream.read(cfg.block_size)
                if overflowed:
                    continue

                arr = self.np.frombuffer(data, dtype=self.np.int16).astype(self.np.float32)
                rms = float(self.np.sqrt(self.np.mean(arr * arr)))

                # Track baseline noise floor conservatively.
                if rms < noise * 2.5:
                    noise = 0.98 * noise + 0.02 * rms

                threshold = max(cfg.abs_floor, noise * cfg.threshold_multiplier)
                now = time.monotonic()

                if rms > threshold and (now - last_spike) > cfg.min_gap_sec:
                    last_spike = now
                    spikes.append(now)
                    spikes = [t for t in spikes if now - t <= cfg.max_double_clap_sec]

                    if len(spikes) >= 2 and (spikes[-1] - spikes[-2]) <= cfg.max_double_clap_sec:
                        return True


class VoiceIO:
    def __init__(self):
        _require_voice_dependencies()
        import numpy as np
        import sounddevice as sd
        import speech_recognition as sr
        import pyttsx3

        self.np = np
        self.sd = sd
        self.sr = sr
        self._speech_lock = threading.Lock()
        self._speech_proc = None
        self._say_voice = self._resolve_say_voice() if sys.platform == "darwin" else None

        self.recognizer = sr.Recognizer()
        self.pyttsx3 = pyttsx3
        self.tts = None

    def _resolve_say_voice(self) -> Optional[str]:
        """Pick a valid macOS `say` voice; return None to use system default."""
        preferred = [v.strip() for v in os.getenv("JARVIS_SAY_VOICE", "Alex,Daniel").split(",") if v.strip()]
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            available = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
            for name in preferred:
                if name in available:
                    return name
        except Exception:
            return None
        return None

    def _speak_with_say(self, clean_text: str):
        """
        Speak via macOS `say` and support interruption via stop_speaking().

        Returns:
            True   — finished naturally (success)
            None   — killed by signal (intentional interrupt, do NOT fall back)
            False  — failed to start or non-zero exit without signal (fall back to pyttsx3)
        """
        cmd = ["say"]
        if self._say_voice:
            cmd.extend(["-v", self._say_voice])
        cmd.append(clean_text)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False

        with self._speech_lock:
            self._speech_proc = proc

        try:
            proc.wait()
            # Negative returncode means killed by signal (SIGKILL / SIGTERM) —
            # this is an intentional interrupt, NOT a failure.
            if proc.returncode < 0:
                return None
            return proc.returncode == 0
        finally:
            with self._speech_lock:
                if self._speech_proc is proc:
                    self._speech_proc = None

    def _create_engine(self):
        engine = self.pyttsx3.init()
        engine.setProperty("rate", 172)
        engine.setProperty("volume", 1.0)

        # Prefer lower, assistant-like local voices when available (macOS).
        voice_keywords = os.getenv("JARVIS_VOICE_KEYWORDS", "alex,daniel").lower().split(",")
        voices = engine.getProperty("voices") or []
        picked_id = None
        for vk in [v.strip() for v in voice_keywords if v.strip()]:
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                vid = (getattr(v, "id", "") or "").lower()
                if vk in name or vk in vid:
                    picked_id = v.id
                    break
            if picked_id:
                break

        if picked_id:
            engine.setProperty("voice", picked_id)

        return engine

    def speak(self, text: str):
        clean = re.sub(r"\s+", " ", self._strip_markdown(text)).strip()
        if not clean:
            return

        # On macOS, prefer `say` for interruptible speech.
        # _speak_with_say returns:
        #   True  → finished naturally       → done, no fallback needed
        #   None  → killed by signal (stop)  → intentional, do NOT fall back to pyttsx3
        #   False → failed to start/run      → fall through to pyttsx3
        if sys.platform == "darwin":
            result = self._speak_with_say(clean)
            if result is not False:   # True (success) or None (interrupted)
                return

        try:
            if self.tts is None:
                self.tts = self._create_engine()
            self.tts.say(clean)
            self.tts.runAndWait()
        except Exception:
            # Re-create engine once if backend drops after prior usage.
            self.tts = self._create_engine()
            self.tts.say(clean)
            self.tts.runAndWait()

    def stop_speaking(self):
        """Interrupt any in-progress speech playback if backend supports it."""
        proc = None
        with self._speech_lock:
            if self._speech_proc is not None:
                proc = self._speech_proc
                self._speech_proc = None
        if proc is not None:
            try:
                proc.kill()  # SIGKILL: immediate, unblockable termination
                proc.wait(timeout=1.0)
            except Exception:
                pass
        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception:
                pass

    def listen(self, duration_sec: float = 6.0, sample_rate: int = 16000) -> Optional[str]:
        """
        Record audio for a short window and transcribe using SpeechRecognition.
        Uses Google recognizer backend via speech_recognition.
        """
        frames = int(duration_sec * sample_rate)
        audio = self.sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
        self.sd.wait()

        rms = float(self.np.sqrt(self.np.mean(audio.astype(self.np.float32) ** 2)))
        if rms < 150.0:
            return None

        raw = audio.tobytes()
        audio_data = self.sr.AudioData(raw, sample_rate=sample_rate, sample_width=2)

        try:
            text = self.recognizer.recognize_google(audio_data)
            return text.strip()
        except self.sr.UnknownValueError:
            return None
        except self.sr.RequestError:
            return None

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)
        text = re.sub(r"\n+", " ", text)
        return text


class JarvisCore:
    def __init__(self):
        settings = load_settings()
        setup_logging(settings)

        self.settings = settings
        self.prompts = load_prompts()
        self.rag_chain = RAGChain(settings)
        self.intent_router = IntentRouter(settings, self.prompts)
        self.session_state = SessionState(settings.sqlite_db_path)

        self.session_id = f"jarvis_{int(time.time())}"
        self.stop_phrases = {
            "stop", "stop jarvis", "jarvis stop", "pause", "exit voice mode"
        }
        self.shutdown_phrases = {
            "shutdown", "shutdown jarvis", "jarvis shutdown", "quit", "goodbye"
        }

    def handle_message(self, user_input: str) -> str:
        session_id = self.session_id

        if is_ticket_flow_active(session_id):
            response = handle_create_ticket(
                session_id=session_id,
                user_message=user_input,
                session_state=self.session_state,
                prompts=self.prompts,
            )
            self.session_state.append_to_history(session_id, "user", user_input)
            self.session_state.append_to_history(session_id, "assistant", response)
            return response

        if is_register_flow_active(session_id):
            response = handle_register_user(
                session_id=session_id,
                user_message=user_input,
                session_state=self.session_state,
                prompts=self.prompts,
            )
            self.session_state.append_to_history(session_id, "user", user_input)
            self.session_state.append_to_history(session_id, "assistant", response)
            return response

        intent_result = self.intent_router.classify(user_input)
        guardrail_response = get_guardrail_response(intent_result, self.prompts)
        if guardrail_response:
            self.session_state.append_to_history(session_id, "user", user_input)
            self.session_state.append_to_history(session_id, "assistant", guardrail_response)
            return guardrail_response

        intent = intent_result.intent

        if intent == "create_ticket":
            response = handle_create_ticket(
                session_id=session_id,
                user_message=user_input,
                session_state=self.session_state,
                prompts=self.prompts,
            )
        elif intent == "check_ticket":
            response = handle_check_ticket(user_input, self.session_state, self.rag_chain)
        elif intent == "check_billing":
            response = handle_check_billing(user_input, self.session_state, session_id=session_id)
        elif intent == "register_user":
            response = handle_register_user(
                session_id=session_id,
                user_message=user_input,
                session_state=self.session_state,
                prompts=self.prompts,
            )
        elif intent == "upgrade_plan":
            response = handle_upgrade_plan(user_input, self.session_state, session_id=session_id)
        elif intent == "close_ticket_by_id":
            response = handle_close_ticket_by_id(user_input, self.session_state)
        elif intent == "close_tickets":
            response = handle_close_tickets(self.session_state)
        elif intent == "list_accounts":
            response = handle_list_accounts(self.session_state)
        else:
            rag_response = self.rag_chain.ask(
                question=user_input,
                session_id=session_id,
                session_state=self.session_state,
            )
            response = rag_response.formatted_answer() if rag_response.is_supported else handle_insufficient_evidence(self.prompts)

        self.session_state.append_to_history(session_id, "user", user_input)
        self.session_state.append_to_history(session_id, "assistant", response)
        return response


def run_jarvis():
    """
    Main runtime loop:
    - wait for clap to wake
    - greet
    - stay active for continuous questions
    - stop on voice command
    """
    _require_voice_dependencies()

    io = VoiceIO()
    detector = ClapDetector()
    core = JarvisCore()

    print("Jarvis is online. Clap twice to activate.")

    while True:
        print("Listening for double clap...")
        woke = detector.wait_for_double_clap(timeout_sec=None)
        if not woke:
            continue

        greeting = "Hey boss, what can I help you with today?"
        print(f"JARVIS: {greeting}")
        io.speak(greeting)

        active = True
        while active:
            print("Speak your command...")
            heard = io.listen(duration_sec=7.0)

            if not heard:
                io.speak("I did not catch that. Please repeat.")
                continue

            text = heard.lower().strip()
            print(f"YOU: {heard}")

            if text in core.shutdown_phrases:
                bye = "Shutting down voice mode. Goodbye boss."
                print(f"JARVIS: {bye}")
                io.speak(bye)
                return

            if text in core.stop_phrases:
                pause_msg = "Voice mode paused. Clap twice when you need me again."
                print(f"JARVIS: {pause_msg}")
                io.speak(pause_msg)
                active = False
                continue

            response = core.handle_message(heard)
            print(f"JARVIS: {response}\n")
            io.speak(response)


if __name__ == "__main__":
    try:
        run_jarvis()
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)
