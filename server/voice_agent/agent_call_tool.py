import os
import re
from typing import Optional, TypedDict, List, Dict, Any, Annotated

# LangChain + Gemini
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph (custom graph, not prebuilt agent)
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode  # ok to use a node; we aren't using create_react_agent

# --- Your existing caller ---
from voice_agent.make_call_with_transcript import make_call_with_transcript_sync

# -------------------------
# 1) Tool wrapper
# -------------------------
PHONE_RE = re.compile(r"\+?\d{8,15}")

# Global variable to store the last full call result
_last_call_result: Optional[Dict[str, Any]] = None
# Global variable to store all call results for multiple calls
_all_call_results: List[Dict[str, Any]] = []

def get_last_call_result() -> Optional[Dict[str, Any]]:
    """Get the last full call result from place_call_with_transcript."""
    return _last_call_result

def get_all_call_results() -> List[Dict[str, Any]]:
    """Get all call results from multiple calls."""
    return _all_call_results

def _normalize_phone(s: str) -> str:
    s = s.strip()
    if PHONE_RE.fullmatch(s):
        return s if s.startswith("+") else f"+{s}"
    m = PHONE_RE.search(s)
    if not m:
        raise ValueError("No valid phone number found in input. Provide E.164 like +14155552671.")
    raw = m.group()
    return raw if raw.startswith("+") else f"+{raw}"

def _extract_phone_numbers(text: str) -> List[str]:
    """Extract all phone numbers from a text string."""
    matches = PHONE_RE.findall(text)
    normalized = []
    for match in matches:
        phone = match if match.startswith("+") else f"+{match}"
        if phone not in normalized:  # Remove duplicates
            normalized.append(phone)
    return normalized

@tool("place_call_with_transcript", return_direct=False)
def place_call_with_transcript(
    phone_or_text: str,
    room_name: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = True,
    transcript_timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Place an outbound call via LiveKit SIP and return a compact summary.
    - phone_or_text: E.164 number or any sentence containing one.
    - room_name / from_number optional passthrough.
    - wait_until_answered True by default.
    - transcript_timeout seconds to wait after call ends.
    """
    global _last_call_result
    try:
        phone = _normalize_phone(phone_or_text)
        result = make_call_with_transcript_sync(
            phone_number=phone,
            room_name=room_name,
            from_number=from_number,
            wait_until_answered=wait_until_answered,
            transcript_timeout=transcript_timeout,
        )

        # Store the full call result for later retrieval
        _last_call_result = result

        call_status = result.get("call_status") or {}
        transcript = result.get("transcript")

        return {
            "ok": True,
            "phone_number": result.get("phone_number"),
            "room_name": result.get("room_name"),
            "dispatch_id": result.get("dispatch_id"),
            "status_outcome": call_status.get("outcome"),
            "answered_at": call_status.get("answered_at"),
            "ended_at": call_status.get("ended_at"),
            "duration_seconds": call_status.get("duration_seconds"),
            "last_status": call_status.get("last_status"),
            "call_attributes": call_status.get("call_attributes", {}),
            "transcript_available": bool(transcript),
            "transcript_file": transcript.get("transcript_file") if transcript else None,
            "transcript_items_count": len(transcript.get("items", [])) if transcript else 0,
            "transcript_preview": (
                transcript.get("items", [])[:5] if transcript and transcript.get("items") else []
            ),
        }
    except Exception as e:
        _last_call_result = None
        return {"ok": False, "error": str(e)}

@tool("place_multiple_calls_with_transcript", return_direct=False)
def place_multiple_calls_with_transcript(
    phone_numbers_text: str,
    room_name_prefix: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = True,
    transcript_timeout: float = 30.0,
    delay_between_calls: float = 0.0,
) -> Dict[str, Any]:
    """
    Place multiple outbound calls via LiveKit SIP and return summaries for all calls.
    - phone_numbers_text: Text containing one or more phone numbers (e.g., "+1234567890 and +9876543210" or "call +123, +456, +789")
    - room_name_prefix / from_number optional passthrough.
    - wait_until_answered True by default.
    - transcript_timeout seconds to wait after each call ends.
    - delay_between_calls seconds to wait between calls (default 0.0).
    """
    global _last_call_result, _all_call_results
    _all_call_results = []  # Reset for new batch
    
    try:
        # Extract all phone numbers from the text
        phone_numbers = _extract_phone_numbers(phone_numbers_text)
        
        if not phone_numbers:
            return {"ok": False, "error": "No valid phone numbers found in the input text."}
        
        results = []
        successful = 0
        failed = 0
        
        for idx, phone in enumerate(phone_numbers):
            try:
                # Add delay between calls if specified and not the first call
                if delay_between_calls > 0 and idx > 0:
                    import time
                    time.sleep(delay_between_calls)
                
                result = make_call_with_transcript_sync(
                    phone_number=phone,
                    room_name=None,  # Let it auto-generate
                    from_number=from_number,
                    wait_until_answered=wait_until_answered,
                    transcript_timeout=transcript_timeout,
                )
                
                # Store each result
                _all_call_results.append(result)
                _last_call_result = result  # Keep last one for backward compatibility
                
                call_status = result.get("call_status") or {}
                transcript = result.get("transcript")
                
                results.append({
                    "ok": True,
                    "phone_number": result.get("phone_number"),
                    "room_name": result.get("room_name"),
                    "dispatch_id": result.get("dispatch_id"),
                    "status_outcome": call_status.get("outcome"),
                    "answered_at": call_status.get("answered_at"),
                    "ended_at": call_status.get("ended_at"),
                    "duration_seconds": call_status.get("duration_seconds"),
                    "last_status": call_status.get("last_status"),
                    "transcript_available": bool(transcript),
                    "transcript_items_count": len(transcript.get("items", [])) if transcript else 0,
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    "ok": False,
                    "phone_number": phone,
                    "error": str(e)
                })
                failed += 1
        
        return {
            "ok": True,
            "total_calls": len(phone_numbers),
            "successful_calls": successful,
            "failed_calls": failed,
            "results": results
        }
    except Exception as e:
        _all_call_results = []
        _last_call_result = None
        return {"ok": False, "error": str(e)}

# -------------------------
# 2) Agent state & prompt
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]

SYSTEM_PROMPT = """You are a concise voice-ops assistant.

When the user asks to make phone calls:
- If the user mentions ONE phone number (e.g., "make call to +91...", "call +1234567890"):
  • Use `place_call_with_transcript` tool with the raw user text as `phone_or_text`.
  
- If the user mentions MULTIPLE phone numbers (e.g., "make call to +91... and +99...", "call +123, +456, +789"):
  • Use `place_multiple_calls_with_transcript` tool with the raw user text containing all numbers as `phone_numbers_text`.
  • The tool will automatically extract and call all numbers found in the text.

After the tool returns, summarize clearly:
  • For single calls: Outcome, Duration, and transcript info if available.
  • For multiple calls: Total calls, successful/failed counts, and brief summary of each call.
If no valid number is present, ask for an E.164 number like +14155552671.
Keep responses brief and actionable.
"""

# -------------------------
# 3) Model node
# -------------------------
def build_llm_with_tools():
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001"),
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    return llm.bind_tools([place_call_with_transcript, place_multiple_calls_with_transcript])

def call_model(state: AgentState) -> Dict[str, Any]:
    llm = build_llm_with_tools()
    # Get messages from state, ensure it's not empty
    existing_messages = state.get("messages", [])
    if not existing_messages:
        raise ValueError("No messages in state. Cannot invoke model with empty message list.")
    
    # Filter out None or invalid messages
    valid_messages = [msg for msg in existing_messages if msg is not None]
    
    if not valid_messages:
        raise ValueError("No valid messages in state. Cannot invoke model.")
    
    # Ensure system prompt is included each turn
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + valid_messages
    
    try:
        ai = llm.invoke(msgs)
    except Exception as e:
        # Debug: print message details if invocation fails
        print(f"[DEBUG] Error invoking model with {len(msgs)} messages")
        print(f"[DEBUG] Message types: {[type(m).__name__ for m in msgs]}")
        print(f"[DEBUG] Message contents preview: {[str(m.content)[:50] if hasattr(m, 'content') else 'N/A' for m in msgs]}")
        raise
    
    # Only return the new AI message - LangGraph will merge it with existing messages
    return {"messages": [ai]}

# -------------------------
# 4) Router: model → tools or end
# -------------------------
def route_after_model(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"

# -------------------------
# 5) Build the graph
# -------------------------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([place_call_with_transcript, place_multiple_calls_with_transcript]))

    graph.set_entry_point("model")
    graph.add_edge("tools", "model")  # after tool results, let the model summarize
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", "end": END})
    return graph.compile()

# -------------------------
# 6) Public helpers you can import
# -------------------------
def run_agent_once(user_text: str) -> Dict[str, Any]:
    """
    Run a single user turn through the custom LangGraph and return both assistant text and call data.
    
    Returns:
        Dictionary with:
        - "response": The assistant's text response
        - "call_result": Full call result (if a single call was made), None otherwise
        - "call_results": List of all call results (if multiple calls were made), empty otherwise
    """
    global _last_call_result, _all_call_results
    _last_call_result = None  # Reset before running
    _all_call_results = []  # Reset before running
    
    app = build_graph()
    state: AgentState = {"messages": [HumanMessage(content=user_text)]}
    out = app.invoke(state)
    # last message is the assistant summary
    print(out)
    
    response_text = out["messages"][-1].content
    call_result = get_last_call_result()
    all_call_results = get_all_call_results()
    
    return {
        "response": response_text,
        "call_result": call_result,
        "call_results": all_call_results
    }

# -------------------------
# 7) CLI quick test
# -------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python agent_call_graph.py "make call to +9102920202"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    result = run_agent_once(query)
    print("\nAgent Response:")
    print(result["response"])
    if result["call_result"]:
        print("\nCall Result:")
        print(f"  Phone: {result['call_result'].get('phone_number')}")
        print(f"  Room: {result['call_result'].get('room_name')}")
        call_status = result['call_result'].get('call_status', {})
        print(f"  Outcome: {call_status.get('outcome')}")
        print(f"  Duration: {call_status.get('duration_seconds')} seconds")
        transcript = result['call_result'].get('transcript')
        if transcript:
            print(f"  Transcript: {len(transcript.get('items', []))} items")
