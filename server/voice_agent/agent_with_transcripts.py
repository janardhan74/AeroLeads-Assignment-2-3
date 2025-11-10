# agent_with_transcripts.py
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
        # Assistant-level guidance (voice style + brand)
        super().__init__(
            instructions=(
                "You are Aeroleads Agent — a friendly, concise voice assistant. "
                "You can help with Aeroleads customer questions AND general knowledge queries. "
                "Keep replies brief, conversational, and easy to speak aloud. "
                "Prefer direct answers over long explanations. "
                "If a fact can change (e.g., leaders, prices, schedules), mention that it’s current as of today. "
                "If you’re unsure, say so rather than guessing. "
                "For Aeroleads support, be helpful and professional; offer next steps when relevant."
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
        # LLM with general-purpose + customer-support system prompt
        llm = google.LLM(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            system_instruction=(
                "ROLE: You are Aeroleads Agent, a voice-first AI that answers BOTH customer support "
                "questions about Aeroleads and general questions (facts, how-tos, definitions, etc.).\n"
                "STYLE: Short, clear, natural speech. Avoid jargon. No long lists; max 3 quick points if needed. "
                "Prefer direct answers.\n"
                "FACTS: For time-sensitive facts (leaders, prices, schedules, laws), say 'As of today' or include the current date. "
                "If uncertain, say you’re not sure rather than inventing details.\n"
                "BOUNDARIES: Don’t provide medical, legal, or financial advice; give general info and suggest consulting a professional. "
                "Don’t collect sensitive data (passwords, full card numbers). "
                "For Aeroleads inquiries, you may briefly describe offerings, pricing approach, or steps to get help.\n"
                "INTERACTION: Be polite. Ask a brief clarifying question only if essential to avoid a wrong answer. "
                "Otherwise, answer with best effort and keep it concise."
            ),
        )
        tts = deepgram.TTS(model=DEEPGRAM_TTS_MODEL, api_key=DEEPGRAM_API_KEY)

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            # use_tts_aligned_transcript=True,  # enable if you want TTS-synced text
        )

        # --- Transcript listeners ---
        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
            print(f"[user STT] final={ev.is_final} lang={ev.language} text={ev.transcript}")

        @session.on("conversation_item_added")
        def _on_conv_item(ev):
            role = getattr(ev.item, "role", "unknown")
            texts = []
            for c in getattr(ev.item, "content", []) or []:
                t = getattr(c, "text", None) or getattr(c, "value", None) or str(c)
                if t:
                    texts.append(t)
            if texts:
                print(f"[history] {role}: {' '.join(texts)}")

        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_output_options=RoomOutputOptions(),
        )

        await session.say("Hi, this is Aeroleads Agent. How can I help you today?")
        await session.generate_reply(instructions="Greet the caller briefly and invite their question.")

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
