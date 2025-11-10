
import asyncio
import os
import datetime
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPMediaEncryption

load_dotenv()

# Environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OUTBOUND_TRUNK_ID = os.getenv("OUTBOUND_TRUNK_ID")
FROM_NUMBER = os.getenv("FROM_NUMBER")
AGENT_NAME = os.getenv("AGENT_NAME", "aeroleads-voice-agent")
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "./transcripts"))
SIP_PARTICIPANT_ID = os.getenv("SIP_PARTICIPANT_ID", "pstn-callee")

# Default values
WAIT_UNTIL_ANSWERED = True
POLL_INTERVAL = 0.8  # seconds between status checks
MAX_CALL_WAIT = 180  # max seconds to wait for call
TRANSCRIPT_WAIT_TIMEOUT = 30  # max seconds to wait for transcript after call ends
TRANSCRIPT_POLL_INTERVAL = 1.0  # seconds between transcript file checks


def _now():
    """Get current UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


def _generate_room_name(phone_number: str) -> str:
    """Generate a unique room name based on phone number and timestamp."""
    # Sanitize phone number for use in room name (remove + and spaces)
    sanitized_phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"call-{sanitized_phone}-{timestamp}"


async def monitor_call(
    lk: api.LiveKitAPI,
    room: str,
    identity: str,
    poll_s: float = POLL_INTERVAL,
    max_wait_s: int = MAX_CALL_WAIT
) -> Dict[str, Any]:
    """
    Poll the room for the SIP participant. Track sip.callStatus and compute duration.
    Returns a dict with outcome + timestamps.
    """
    answered_at = None
    last_status = None
    first_seen = None
    call_attributes = {}
    t0 = time.monotonic()

    while True:
        # Fetch current participant list
        parts_resp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        p = next((pp for pp in parts_resp.participants if pp.identity == identity), None)

        if p is None:
            # Participant left the room
            if answered_at:
                return {
                    "outcome": "completed",
                    "answered_at": answered_at.isoformat() if answered_at else None,
                    "ended_at": _now().isoformat(),
                    "duration_seconds": int((_now() - answered_at).total_seconds()) if answered_at else 0,
                    "last_status": last_status,
                    "call_attributes": call_attributes,
                }
            else:
                # Never answered
                return {
                    "outcome": "no_answer_or_declined",
                    "answered_at": None,
                    "ended_at": _now().isoformat(),
                    "duration_seconds": 0,
                    "last_status": last_status,
                    "call_attributes": call_attributes,
                }

        # Log interesting attributes while present
        attrs = getattr(p, "attributes", {}) or {}
        status = attrs.get("sip.callStatus")  # active / dialing / ringing / hangup / automation
        
        if first_seen is None:
            first_seen = _now()
            # Store call attributes for later use
            call_attributes = {
                "sip.callID": attrs.get("sip.callID"),
                "sip.callIDFull": attrs.get("sip.callIDFull"),
                "sip.twilio.callSid": attrs.get("sip.twilio.callSid"),
                "sip.phoneNumber": attrs.get("sip.phoneNumber"),
                "sip.trunkID": attrs.get("sip.trunkID"),
            }
            print(f"[Call Monitor] SIP attributes: {call_attributes}")

        if status != last_status:
            print(f"[Call Monitor] [{_now().isoformat()}] sip.callStatus → {status}")
            last_status = status

        # When status flips to active, mark as answered
        if status == "active" and answered_at is None:
            answered_at = _now()
            print(f"[Call Monitor] Answered at: {answered_at.isoformat()}")

        # Stop if we've waited too long overall (safety)
        if time.monotonic() - t0 > max_wait_s:
            return {
                "outcome": "timeout",
                "answered_at": answered_at.isoformat() if answered_at else None,
                "ended_at": _now().isoformat(),
                "duration_seconds": int((_now() - (answered_at or first_seen)).total_seconds()) if (answered_at or first_seen) else 0,
                "last_status": last_status,
                "call_attributes": call_attributes,
            }

        await asyncio.sleep(poll_s)


async def wait_for_transcript(
    room_name: str,
    call_start_time: Optional[datetime.datetime] = None,
    timeout: float = TRANSCRIPT_WAIT_TIMEOUT,
    poll_interval: float = TRANSCRIPT_POLL_INTERVAL
) -> Optional[Dict[str, Any]]:
    """
    Wait for transcript file to appear after call ends.
    Transcript files are named: transcript_{room_name}_{timestamp}.json
    
    Args:
        room_name: The room name used for the call
        call_start_time: Optional datetime when the call started (to filter old files)
        timeout: Maximum seconds to wait for transcript
        poll_interval: Seconds between file system checks
    
    Returns:
        Transcript data as dict, or None if not found within timeout
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.monotonic()
    # Use provided call_start_time or default to a reasonable time before now
    if call_start_time is None:
        call_start_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=timeout + 60)
    elif isinstance(call_start_time, datetime.datetime):
        # Ensure timezone-aware
        if call_start_time.tzinfo is None:
            call_start_time = call_start_time.replace(tzinfo=datetime.timezone.utc)
    
    print(f"[Transcript] Waiting for transcript for room: {room_name}")
    print(f"[Transcript] Searching in directory: {TRANSCRIPTS_DIR}")
    print(f"[Transcript] Looking for files created after: {call_start_time.isoformat()}")
    
    # Track files we've already checked to avoid re-reading
    checked_files = set()
    
    while time.monotonic() - start_time < timeout:
        # Look for transcript files matching the room name
        # The file pattern is: transcript_{room_name}_{timestamp}.json
        pattern = f"transcript_{room_name}_*.json"
        matching_files = list(TRANSCRIPTS_DIR.glob(pattern))
        
        # Also check for files with AGENT_NAME prefix as fallback
        agent_pattern = f"transcript_{AGENT_NAME}_*.json"
        agent_files = list(TRANSCRIPTS_DIR.glob(agent_pattern))
        
        # Combine and deduplicate
        all_files = list(set(matching_files + agent_files))
        
        # Filter to files created after call start and not yet checked
        new_files = [
            f for f in all_files
            if str(f) not in checked_files
            and datetime.datetime.fromtimestamp(f.stat().st_mtime, tz=datetime.timezone.utc) >= call_start_time
        ]
        
        if new_files:
            # Sort by modification time, get the most recent one
            new_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for transcript_file in new_files:
                checked_files.add(str(transcript_file))
                try:
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        transcript_data = json.load(f)
                    
                    # Verify this transcript has content (items list)
                    items = transcript_data.get("items", [])
                    if items:
                        print(f"[Transcript] Found transcript: {transcript_file}")
                        print(f"[Transcript] Transcript contains {len(items)} conversation items")
                        return {
                            "transcript_file": str(transcript_file),
                            "transcript_data": transcript_data,
                            "items": items,
                        }
                except Exception as e:
                    print(f"[Transcript] Error reading transcript file {transcript_file}: {e}")
        
        await asyncio.sleep(poll_interval)
    
    print(f"[Transcript] Timeout waiting for transcript (waited {timeout}s)")
    print(f"[Transcript] Checked files: {len(checked_files)}")
    return None


async def make_call_with_transcript(
    phone_number: str,
    room_name: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = WAIT_UNTIL_ANSWERED,
    transcript_timeout: float = TRANSCRIPT_WAIT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Make an outbound call and collect both call status and transcript.
    
    Args:
        phone_number: Phone number to dial (E.164 format, e.g., +1234567890)
        room_name: Optional room name. If not provided, a unique name is generated.
        from_number: Optional caller ID number
        wait_until_answered: Whether to wait until the call is answered
        transcript_timeout: Maximum seconds to wait for transcript after call ends
    
    Returns:
        Dictionary containing:
        - call_status: Call status information (outcome, duration, timestamps, etc.)
        - transcript: Transcript data if available (None if not found)
        - room_name: Room name used for the call
        - phone_number: Phone number that was dialed
    
    Raises:
        ValueError: If required environment variables are missing
        Exception: If call creation fails
    """
    # Validate required environment variables
    missing = []
    for key, value in {
        "LIVEKIT_URL": LIVEKIT_URL,
        "LIVEKIT_API_KEY": LIVEKIT_API_KEY,
        "LIVEKIT_API_SECRET": LIVEKIT_API_SECRET,
        "OUTBOUND_TRUNK_ID": OUTBOUND_TRUNK_ID,
    }.items():
        if not value:
            missing.append(key)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Generate room name if not provided
    if not room_name:
        room_name = _generate_room_name(phone_number)
    
    # Use provided from_number or fall back to env var
    caller_id = from_number or FROM_NUMBER
    
    print(f"[Make Call] Dialing {phone_number} into room '{room_name}'")
    
    # Convert wss:// URL to https:// for REST API if needed
    def _rest_base(url: str) -> str:
        if url.startswith("wss://"):
            return "https://" + url[len("wss://"):]
        return url
    
    api_url = _rest_base(LIVEKIT_URL)
    
    # Initialize LiveKit API
    lk = api.LiveKitAPI(
        url=api_url,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )
    
    dispatch_id = None
    # Record call start time for transcript lookup
    call_start_timestamp = _now()
    
    try:
        # Step 1: Create agent dispatch so the agent joins the room
        print(f"[Make Call] Creating agent dispatch for room '{room_name}'...")
        dispatch = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps({"reason": "outbound_call", "phone_number": phone_number}),
            )
        )
        dispatch_id = dispatch.id
        print(f"[Make Call] Dispatch created: {dispatch_id}")
        
        # Step 2: Wait a moment for the agent to join the room
        print(f"[Make Call] Waiting for agent to join room...")
        await asyncio.sleep(2)
        
        # Step 3: Create SIP participant request
        req = CreateSIPParticipantRequest(
            sip_trunk_id=OUTBOUND_TRUNK_ID,
            sip_call_to=phone_number,
            room_name=room_name,
            participant_identity=SIP_PARTICIPANT_ID,
            participant_name="Phone Callee",
            wait_until_answered=wait_until_answered,
            media_encryption=SIPMediaEncryption.SIP_MEDIA_ENCRYPT_ALLOW,
        )
        
        if caller_id:
            req.sip_number = caller_id
        
        # Step 4: Create SIP participant (this will dial the number)
        print(f"[Make Call] Creating SIP participant...")
        part = await lk.sip.create_sip_participant(req)
        print(f"[Make Call] SIP participant created: {part.participant_id}")
        
        # If wait_until_answered=True, we reach here only after the user picked up.
        # Monitor the call until it ends and collect status
        print(f"[Make Call] Monitoring call status...")
        call_status = await monitor_call(lk, room_name, SIP_PARTICIPANT_ID)
        print(f"[Make Call] Call ended. Status: {call_status['outcome']}")
        
        # Wait for transcript to be generated by the agent
        # Note: The transcript file is saved when the agent job ends, which happens after the call ends
        print(f"[Make Call] Waiting for transcript (call started at {call_start_timestamp.isoformat()})...")
        transcript = await wait_for_transcript(
            room_name,
            call_start_time=call_start_timestamp,
            timeout=transcript_timeout
        )
        
        # Prepare result
        result = {
            "phone_number": phone_number,
            "room_name": room_name,
            "dispatch_id": dispatch_id,
            "call_status": call_status,
            "transcript": transcript,
        }
        
        return result
        
    except Exception as e:
        print(f"[Make Call] Error: {e}")
        raise
    finally:
        await lk.aclose()


# Convenience function for synchronous usage
def make_call_with_transcript_sync(
    phone_number: str,
    room_name: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = WAIT_UNTIL_ANSWERED,
    transcript_timeout: float = TRANSCRIPT_WAIT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for make_call_with_transcript.
    
    Usage:
        result = make_call_with_transcript_sync("+1234567890")
        print(result["call_status"])
        print(result["transcript"])
    """
    return asyncio.run(
        make_call_with_transcript(
            phone_number=phone_number,
            room_name=room_name,
            from_number=from_number,
            wait_until_answered=wait_until_answered,
            transcript_timeout=transcript_timeout,
        )
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python make_call_with_transcript.py <phone_number>")
        print("Example: python make_call_with_transcript.py +1234567890")
        sys.exit(1)
    
    phone = sys.argv[1]
    print(f"Making call to {phone}...")
    
    try:
        result = make_call_with_transcript_sync(phone)
        
        print("\n" + "="*50)
        print("CALL RESULT")
        print("="*50)
        print(f"Phone Number: {result['phone_number']}")
        print(f"Room Name: {result['room_name']}")
        print(f"\nCall Status:")
        for key, value in result['call_status'].items():
            print(f"  {key}: {value}")
        
        if result['transcript']:
            print(f"\nTranscript: Available")
            print(f"  File: {result['transcript']['transcript_file']}")
            print(f"  Items: {len(result['transcript']['items'])} conversation items")
        else:
            print(f"\nTranscript: Not available (timeout or not generated)")
        
    except Exception as e:
        print(f"Error making call: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

