# voice_agent_routes.py
from __future__ import annotations

import sys
import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor

# Add server directory to path for imports (to allow voice_agent imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from voice_agent package
from voice_agent.make_call_with_transcript import make_call_with_transcript, make_call_with_transcript_sync
from voice_agent.agent_call_tool import run_agent_once
from voice_agent.batch_call import batch_call_phone_numbers

# Create router for voice agent routes
router = APIRouter(prefix="/api/voice", tags=["voice-agent"])


def _make_json_serializable(obj):
    """Recursively convert non-serializable objects to serializable format."""
    import datetime
    import json
    from decimal import Decimal
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(key): _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_json_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class MakeCallRequest(BaseModel):
    phone_number: str
    room_name: Optional[str] = None
    from_number: Optional[str] = None
    wait_until_answered: bool = True
    transcript_timeout: float = 30.0


class MakeAgentCallRequest(BaseModel):
    prompt: str


class MakeMultipleCallsRequest(BaseModel):
    phone_numbers: List[str] = Field(..., min_length=1, description="List of phone numbers to call")
    room_name_prefix: Optional[str] = None
    from_number: Optional[str] = None
    wait_until_answered: bool = True
    transcript_timeout: float = 30.0
    delay_between_calls: float = 0.0


@router.post("/make-a-call")
async def make_a_call(request: MakeCallRequest):
    """
    Make a single phone call using make_call_with_transcript (async version).
    
    Args:
        request: MakeCallRequest containing phone_number and optional parameters
    
    Returns:
        Complete response from make_call_with_transcript function
    """
    try:
        # Use async version directly since we're in an async route handler
        result = await make_call_with_transcript(
            phone_number=request.phone_number,
            room_name=request.room_name,
            from_number=request.from_number,
            wait_until_answered=request.wait_until_answered,
            transcript_timeout=request.transcript_timeout,
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/make-agent-call")
async def make_agent_call(request: MakeAgentCallRequest):
    """
    Run the voice agent with a user prompt using run_agent_once.
    
    Args:
        request: MakeAgentCallRequest containing the user prompt
    
    Returns:
        Response similar to make-call API with:
        - prompt: The original prompt
        - response: The agent's text response
        - phone_number, room_name, dispatch_id: Call details (if call was made)
        - call_status: Call status information (if call was made)
        - transcript: Transcript data (if call was made and transcript available)
        - success: Boolean indicating if the request was successful
    """
    print(f"[make-agent-call] Starting request with prompt: {request.prompt}")
    try:
        if not request.prompt or not request.prompt.strip():
            print("[make-agent-call] ERROR: Empty prompt")
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        print("[make-agent-call] Prompt validated, preparing to call run_agent_once")
        
        # Run in thread pool executor to avoid event loop conflicts
        # since run_agent_once may call functions that use asyncio.run()
        loop = asyncio.get_event_loop()
        print("[make-agent-call] Got event loop, creating ThreadPoolExecutor")
        
        with ThreadPoolExecutor() as executor:
            print("[make-agent-call] Executor created, calling run_agent_once in executor")
            agent_result = await loop.run_in_executor(
                executor,
                run_agent_once,
                request.prompt
            )
        
        print(f"[make-agent-call] run_agent_once completed")
        print(f"[make-agent-call] agent_result type: {type(agent_result)}")
        print(f"[make-agent-call] agent_result keys: {agent_result.keys() if isinstance(agent_result, dict) else 'N/A'}")
        
        # Extract response text and call result
        response_text = agent_result.get("response", "") if isinstance(agent_result, dict) else str(agent_result)
        call_result = agent_result.get("call_result") if isinstance(agent_result, dict) else None
        call_results = agent_result.get("call_results", []) if isinstance(agent_result, dict) else []
        
        # Build response similar to make-call API format
        response_dict = {
            "prompt": request.prompt,
            "response": response_text,
            "success": True
        }
        
        # If multiple calls were made, include all results
        if call_results and len(call_results) > 1:
            response_dict["total_calls"] = len(call_results)
            response_dict["call_results"] = call_results
            # Also include summary stats
            successful = sum(1 for r in call_results if r.get("call_status", {}).get("outcome") == "completed")
            response_dict["successful_calls"] = successful
            response_dict["failed_calls"] = len(call_results) - successful
        # If a single call was made, include call details and transcript in the same format as make-call
        elif call_result:
            response_dict["phone_number"] = call_result.get("phone_number")
            response_dict["room_name"] = call_result.get("room_name")
            response_dict["dispatch_id"] = call_result.get("dispatch_id")
            response_dict["call_status"] = call_result.get("call_status")
            response_dict["transcript"] = call_result.get("transcript")
        
        print(f"[make-agent-call] Created response_dict with call_result: {call_result is not None}")
        
        # Make sure all datetime objects and other non-serializable objects are converted
        response_dict = _make_json_serializable(response_dict)
        
        print("[make-agent-call] Creating JSONResponse...")
        json_response = JSONResponse(content=response_dict)
        print("[make-agent-call] JSONResponse created successfully")
        print(f"[make-agent-call] Returning response")
        return json_response
    
    except HTTPException as he:
        print(f"[make-agent-call] HTTPException raised: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        import traceback
        print(f"[make-agent-call] Exception caught: {type(e).__name__}: {str(e)}")
        print(f"[make-agent-call] Full traceback:\n{traceback.format_exc()}")
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/make-multiple-calls")
async def make_multiple_calls(request: MakeMultipleCallsRequest):
    """
    Make multiple phone calls using batch_call_phone_numbers.
    
    Args:
        request: MakeMultipleCallsRequest containing list of phone numbers and optional parameters
    
    Returns:
        Complete response from batch_call_phone_numbers function
    """
    try:
        if not request.phone_numbers or len(request.phone_numbers) == 0:
            raise HTTPException(status_code=400, detail="phone_numbers list cannot be empty")
        
        # Create a wrapper function to pass all arguments
        def call_batch_function():
            return batch_call_phone_numbers(
                phone_numbers=request.phone_numbers,
                max_workers=1,  # Sequential calls as specified
                room_name_prefix=request.room_name_prefix,
                from_number=request.from_number,
                wait_until_answered=request.wait_until_answered,
                transcript_timeout=request.transcript_timeout,
                delay_between_calls=request.delay_between_calls,
            )
        
        # Run the sync batch_call_phone_numbers in a thread pool to avoid event loop conflicts
        # since it uses asyncio.run() internally
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, call_batch_function)
        
        # Ensure all datetime objects and other non-serializable objects are converted to strings
        # The result should already have ISO format strings, but convert any remaining issues
        result = _make_json_serializable(result)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

