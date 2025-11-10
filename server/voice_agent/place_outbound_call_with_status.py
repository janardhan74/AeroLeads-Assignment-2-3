import asyncio, os, datetime, time
from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPMediaEncryption

load_dotenv()

LIVEKIT_URL       = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY   = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET= os.getenv("LIVEKIT_API_SECRET")
OUTBOUND_TRUNK_ID = os.getenv("OUTBOUND_TRUNK_ID")
ROOM_NAME         = os.getenv("ROOM_NAME", "phone-demo")
DIAL_NUMBER       = os.getenv("DIAL_NUMBER")
FROM_NUMBER       = os.getenv("FROM_NUMBER")
IDENTITY          = os.getenv("SIP_PARTICIPANT_ID", "pstn-callee")

WAIT_UNTIL_ANSWERED = True  # keeps create_sip_participant blocked until the callee answers

def _now():
    return datetime.datetime.now(datetime.timezone.utc)

async def monitor_call(lk: api.LiveKitAPI, room: str, identity: str, poll_s: float = 0.8, max_wait_s: int = 180):
    """
    Poll the room for the SIP participant. Track sip.callStatus and compute duration.
    Returns a dict with outcome + timestamps.
    """
    answered_at = None
    last_status = None
    first_seen  = None
    t0 = time.monotonic()

    while True:
        # fetch current participant list
        parts_resp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        p = next((pp for pp in parts_resp.participants if pp.identity == identity), None)

        if p is None:
            # participant left the room
            if answered_at:
                return {
                    "outcome": "completed",
                    "answered_at": answered_at,
                    "ended_at": _now(),
                    "duration_seconds": int((_now() - answered_at).total_seconds()),
                    "last_status": last_status,
                }
            else:
                # never answered—cannot distinguish busy/decline/no-answer without webhooks
                return {
                    "outcome": "no_answer_or_declined",
                    "answered_at": None,
                    "ended_at": _now(),
                    "duration_seconds": 0,
                    "last_status": last_status,
                }

        # log interesting attributes while present
        attrs = getattr(p, "attributes", {}) or {}
        status = attrs.get("sip.callStatus")  # active / dialing / ringing / hangup / automation
        if first_seen is None:
            first_seen = _now()
            # these are useful for correlating with provider logs
            print("SIP attributes snapshot:", {
                "sip.callID": attrs.get("sip.callID"),
                "sip.callIDFull": attrs.get("sip.callIDFull"),
                "sip.twilio.callSid": attrs.get("sip.twilio.callSid"),
                "sip.phoneNumber": attrs.get("sip.phoneNumber"),
                "sip.trunkID": attrs.get("sip.trunkID"),
            })

        if status != last_status:
            print(f"[{_now().isoformat()}] sip.callStatus → {status}")
            last_status = status

        # when status flips to active, mark as answered
        if status == "active" and answered_at is None:
            answered_at = _now()
            print(f"Answered at: {answered_at.isoformat()}")

        # stop if we’ve waited too long overall (safety)
        if time.monotonic() - t0 > max_wait_s:
            return {
                "outcome": "timeout",
                "answered_at": answered_at,
                "ended_at": _now(),
                "duration_seconds": int((_now() - (answered_at or first_seen)).total_seconds()) if (answered_at or first_seen) else 0,
                "last_status": last_status,
            }

        await asyncio.sleep(poll_s)

async def main():
    lk = api.LiveKitAPI(url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    try:
        req = CreateSIPParticipantRequest(
            sip_trunk_id=OUTBOUND_TRUNK_ID,
            sip_call_to=DIAL_NUMBER,
            room_name=ROOM_NAME,
            participant_identity=IDENTITY,
            participant_name="Phone Callee",
            wait_until_answered=WAIT_UNTIL_ANSWERED,
            media_encryption=SIPMediaEncryption.SIP_MEDIA_ENCRYPT_ALLOW,
        )
        if FROM_NUMBER:
            req.sip_number = FROM_NUMBER

        print(f"Dialing {DIAL_NUMBER} via trunk {OUTBOUND_TRUNK_ID} into room '{ROOM_NAME}' ...")
        part = await lk.sip.create_sip_participant(req)
        print("Created SIP participant:", part.participant_id)

        # If wait_until_answered=True, we reach here only after the user picked up.
        # Monitor until hangup and compute duration.
        result = await monitor_call(lk, ROOM_NAME, IDENTITY)
        print("=== Call result ===")
        for k, v in result.items():
            print(f"{k}: {v}")

    except Exception as e:
        print("Error:", e)
    finally:
        await lk.aclose()

if __name__ == "__main__":
    missing = [k for k, v in {
        "LIVEKIT_URL": LIVEKIT_URL,
        "LIVEKIT_API_KEY": LIVEKIT_API_KEY,
        "LIVEKIT_API_SECRET": LIVEKIT_API_SECRET,
        "OUTBOUND_TRUNK_ID": OUTBOUND_TRUNK_ID,
        "DIAL_NUMBER": DIAL_NUMBER,
    }.items() if not v]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))
    asyncio.run(main())