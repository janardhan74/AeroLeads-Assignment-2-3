from dotenv import load_dotenv
import os, asyncio, json
from datetime import datetime
from pathlib import Path

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomOutputOptions
from livekit.plugins import deepgram, google

# Events API (for callback typing; optional but nice)
from livekit.agents import UserInputTranscribedEvent

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "aeroleads-voice-agent")

GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_STT_MODEL  = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_TTS_MODEL  = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY", "")

TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "./transcripts"))
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a concise, friendly voice assistant. "
                "Keep replies brief and conversational."
            )
        )

async def entrypoint(ctx: agents.JobContext):
    session: AgentSession | None = None

    async def save_transcript():
        """Write the full conversation history to disk (JSON) at shutdown."""
        if session is None:
            return
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            room_name = getattr(ctx.room, "name", "unknown-room")
            out_path = TRANSCRIPTS_DIR / f"transcript_{room_name}_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(session.history.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"[transcripts] saved: {out_path}")
        except Exception as e:
            print("[transcripts] save failed:", e)

    # ensure we always persist the transcript
    ctx.add_shutdown_callback(save_transcript)

    try:
        # I/O models
        stt = deepgram.STT(
            model=DEEPGRAM_STT_MODEL,
            punctuate=True,
            interim_results=True,
            api_key=DEEPGRAM_API_KEY,
        )
        llm = google.LLM(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
        tts = deepgram.TTS(model=DEEPGRAM_TTS_MODEL, api_key=DEEPGRAM_API_KEY)

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            # Optional: better text sync for live displays (disable sync if you want ASAP text)
            # use_tts_aligned_transcript=True,
        )

        # --- Transcript listeners ---
        # 1) Realtime user transcripts (PSTN side), both interim and final
        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
            # ev.is_final tells you if this chunk is final
            print(f"[user STT] final={ev.is_final} lang={ev.language} text={ev.transcript}")

        # 2) Every committed item in the conversation history (user + assistant)
        @session.on("conversation_item_added")
        def _on_conv_item(ev):
            # ev.item.role in {"user","assistant"}, ev.item.content is list of text chunks, we can flatten
            role = getattr(ev.item, "role", "unknown")
            texts = []
            for c in getattr(ev.item, "content", []) or []:
                # text content in .text, depending on SDK version; keep defensive
                t = getattr(c, "text", None) or getattr(c, "value", None) or str(c)
                if t:
                    texts.append(t)
            if texts:
                print(f"[history] {role}: {' '.join(texts)}")

        # Start session with transcription output enabled (default True). You can also set
        # sync_transcription=False to emit text as soon as available rather than synced to TTS.
        await session.start(
            room=ctx.room,  # pass the room object
            agent=Assistant(),
            room_output_options=RoomOutputOptions(
                # transcription_enabled=True,  # default True
                # text_enabled=True,          # default True
                # sync_transcription=False,   # uncomment for ASAP text (not synced to TTS timing)
            ),
        )

        # Initial greeting so we also capture assistant side
        await session.say("Hi! This is the AeroLeads voice agent. How can I help you today?")
        await session.generate_reply(instructions="Greet the caller briefly and offer help.")

        # keep alive until the call ends / job stops
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("[agent] job cancelled; exiting")
        raise
    except Exception as e:
        print("[agent] error:", e)
    finally:
        if session is not None:
            try:
                await session.aclose()
            except Exception as e:
                print("[agent] session close error:", e)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
            # port=8091,
        )
    )