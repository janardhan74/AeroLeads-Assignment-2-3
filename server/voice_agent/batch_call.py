
import asyncio
import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


try:
    from .make_call_with_transcript import make_call_with_transcript_sync
except ImportError:
    from make_call_with_transcript import make_call_with_transcript_sync


def batch_call_phone_numbers(
    phone_numbers: List[str],
    max_workers: int = 1,
    room_name_prefix: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = True,
    transcript_timeout: float = 30.0,
    delay_between_calls: float = 0.0,
) -> Dict[str, Any]:
    """
    Efficiently call multiple phone numbers and collect all responses.
    
    Args:
        phone_numbers: List of phone numbers in E.164 format (e.g., ["+1234567890", "+0987654321"])
        max_workers: Maximum number of concurrent calls (default: 1 for sequential)
                     Set to a higher value for parallel calls (e.g., 3-5)
        room_name_prefix: Optional prefix for room names (auto-generated if not provided)
        from_number: Optional caller ID number
        wait_until_answered: Whether to wait until each call is answered
        transcript_timeout: Maximum seconds to wait for transcript after call ends
        delay_between_calls: Delay in seconds between starting calls (for sequential mode)
    
    Returns:
        Dictionary containing:
        - total_calls: Total number of phone numbers provided
        - successful_calls: Number of successful calls (no exceptions)
        - failed_calls: Number of failed calls
        - results: List of call results, each containing:
          - phone_number: The phone number that was called
          - success: Boolean indicating if the call succeeded
          - result: The full result from make_call_with_transcript_sync (if successful)
          - error: Error message (if failed)
          - call_index: Index of the call in the input list
          - start_time: Timestamp when the call started
          - end_time: Timestamp when the call completed
          - duration_seconds: Time taken for the call
    
    Example:
        phone_numbers = ["+1234567890", "+0987654321", "+1122334455"]
        results = batch_call_phone_numbers(phone_numbers, max_workers=2)
        
        for result in results["results"]:
            if result["success"]:
                print(f"Call to {result['phone_number']}: {result['result']['call_status']['outcome']}")
            else:
                print(f"Call to {result['phone_number']} failed: {result['error']}")
    """
    if not phone_numbers:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "results": [],
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration_seconds": 0,
        }
    
    start_time = datetime.datetime.now()
    results = []
    successful_count = 0
    failed_count = 0
    
    def make_single_call(phone_number: str, index: int) -> Dict[str, Any]:
        """Make a single call and return result with metadata."""
        call_start = datetime.datetime.now()
        call_result = {
            "phone_number": phone_number,
            "call_index": index,
            "start_time": call_start.isoformat(),
            "success": False,
            "result": None,
            "error": None,
            "end_time": None,
            "duration_seconds": None,
        }
        
        try:
            # Add delay if specified (for sequential calls)
            if delay_between_calls > 0 and index > 0:
                time.sleep(delay_between_calls)
            
            # Generate room name if prefix is provided
            room_name = None
            if room_name_prefix:
                # Generate room name similar to make_call_with_transcript
                sanitized_phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
                timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                room_name = f"{room_name_prefix}-call-{sanitized_phone}-{timestamp}"
            
            # Make the call
            result = make_call_with_transcript_sync(
                phone_number=phone_number,
                room_name=room_name,
                from_number=from_number,
                wait_until_answered=wait_until_answered,
                transcript_timeout=transcript_timeout,
            )
            
            call_end = datetime.datetime.now()
            call_result["success"] = True
            call_result["result"] = result
            call_result["end_time"] = call_end.isoformat()
            call_result["duration_seconds"] = (call_end - call_start).total_seconds()
            
            print(f"[Batch Call] ✓ Call {index + 1}/{len(phone_numbers)} to {phone_number} completed successfully")
            
        except Exception as e:
            call_end = datetime.datetime.now()
            call_result["success"] = False
            call_result["error"] = str(e)
            call_result["end_time"] = call_end.isoformat()
            call_result["duration_seconds"] = (call_end - call_start).total_seconds()
            
            print(f"[Batch Call] ✗ Call {index + 1}/{len(phone_numbers)} to {phone_number} failed: {str(e)}")
        
        return call_result
    
    # Execute calls
    if max_workers == 1:
        # Sequential execution (one at a time)
        print(f"[Batch Call] Starting sequential calls to {len(phone_numbers)} phone number(s)...")
        for index, phone_number in enumerate(phone_numbers):
            result = make_single_call(phone_number, index)
            results.append(result)
            if result["success"]:
                successful_count += 1
            else:
                failed_count += 1
    else:
        # Parallel execution with thread pool
        print(f"[Batch Call] Starting parallel calls to {len(phone_numbers)} phone number(s) with {max_workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all calls
            future_to_phone = {
                executor.submit(make_single_call, phone_number, index): phone_number
                for index, phone_number in enumerate(phone_numbers)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_phone):
                result = future.result()
                results.append(result)
                if result["success"]:
                    successful_count += 1
                else:
                    failed_count += 1
    
    # Sort results by call_index to maintain order
    results.sort(key=lambda x: x["call_index"])
    
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    return {
        "total_calls": len(phone_numbers),
        "successful_calls": successful_count,
        "failed_calls": failed_count,
        "results": results,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
    }


def batch_call_phone_numbers_async(
    phone_numbers: List[str],
    max_concurrent: int = 3,
    room_name_prefix: Optional[str] = None,
    from_number: Optional[str] = None,
    wait_until_answered: bool = True,
    transcript_timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Asynchronously call multiple phone numbers using asyncio for better efficiency.
    
    This is an alternative implementation using asyncio instead of threads.
    It's more efficient for I/O-bound operations like API calls.
    
    Args:
        phone_numbers: List of phone numbers in E.164 format
        max_concurrent: Maximum number of concurrent calls (default: 3)
        room_name_prefix: Optional prefix for room names
        from_number: Optional caller ID number
        wait_until_answered: Whether to wait until each call is answered
        transcript_timeout: Maximum seconds to wait for transcript after call ends
    
    Returns:
        Same structure as batch_call_phone_numbers
    """
    # Try relative import first (when imported as module), fall back to absolute (when run directly)
    try:
        from .make_call_with_transcript import make_call_with_transcript
    except ImportError:
        from make_call_with_transcript import make_call_with_transcript
    
    async def make_single_call_async(phone_number: str, index: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Make a single call asynchronously."""
        call_start = datetime.datetime.now()
        call_result: Dict[str, Any] = {
            "phone_number": phone_number,
            "call_index": index,
            "start_time": call_start.isoformat(),
            "success": False,
            "result": None,
            "error": None,
            "end_time": None,
            "duration_seconds": None,
        }
        
        async with semaphore:
            try:
                # Generate room name if prefix is provided
                room_name = None
                if room_name_prefix:
                    # Generate room name similar to make_call_with_transcript
                    sanitized_phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
                    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    room_name = f"{room_name_prefix}-call-{sanitized_phone}-{timestamp}"
                
                # Make the call
                result = await make_call_with_transcript(
                    phone_number=phone_number,
                    room_name=room_name,
                    from_number=from_number,
                    wait_until_answered=wait_until_answered,
                    transcript_timeout=transcript_timeout,
                )
                
                call_end = datetime.datetime.now()
                call_result["success"] = True
                call_result["result"] = result
                call_result["end_time"] = call_end.isoformat()
                call_result["duration_seconds"] = (call_end - call_start).total_seconds()
                
                print(f"[Batch Call Async] ✓ Call {index + 1}/{len(phone_numbers)} to {phone_number} completed successfully")
                
            except Exception as e:
                call_end = datetime.datetime.now()
                call_result["success"] = False
                call_result["error"] = str(e)
                call_result["end_time"] = call_end.isoformat()
                call_result["duration_seconds"] = (call_end - call_start).total_seconds()
                
                print(f"[Batch Call Async] ✗ Call {index + 1}/{len(phone_numbers)} to {phone_number} failed: {str(e)}")
        
        return call_result
    
    async def run_all_calls():
        """Run all calls concurrently with semaphore limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            make_single_call_async(phone_number, index, semaphore)
            for index, phone_number in enumerate(phone_numbers)
        ]
        return await asyncio.gather(*tasks)
    
    if not phone_numbers:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "results": [],
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration_seconds": 0,
        }
    
    print(f"[Batch Call Async] Starting async calls to {len(phone_numbers)} phone number(s) with max {max_concurrent} concurrent...")
    start_time = datetime.datetime.now()
    
    # Run all calls
    results = asyncio.run(run_all_calls())
    
    # Sort results by call_index to maintain order
    results.sort(key=lambda x: x["call_index"])
    
    # Count successes and failures
    successful_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - successful_count
    
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    return {
        "total_calls": len(phone_numbers),
        "successful_calls": successful_count,
        "failed_calls": failed_count,
        "results": results,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Example phone numbers (replace with actual numbers)
    example_phones = [
        "+917396257294",
        # Add more phone numbers here
    ]
    
    if len(sys.argv) > 1:
        # Use phone numbers from command line arguments
        phone_numbers = sys.argv[1:]
    else:
        # Use example phone numbers
        phone_numbers = example_phones
        print("Using example phone numbers. Provide phone numbers as command line arguments to use custom ones.")
        print(f"Example: python batch_call.py +1234567890 +0987654321")
    
    print(f"\n{'='*60}")
    print(f"BATCH CALLING {len(phone_numbers)} PHONE NUMBER(S)")
    print(f"{'='*60}\n")
    
    # Option 1: Sequential calls (safer, one at a time)
    print("Method 1: Sequential calls (max_workers=1)")
    print("-" * 60)
    result_sequential = batch_call_phone_numbers(
        phone_numbers=phone_numbers,
        max_workers=1,  # Sequential
        delay_between_calls=1.0,  # 1 second delay between calls
    )
    
    print(f"\nSequential Results:")
    print(f"  Total calls: {result_sequential['total_calls']}")
    print(f"  Successful: {result_sequential['successful_calls']}")
    print(f"  Failed: {result_sequential['failed_calls']}")
    print(f"  Total duration: {result_sequential['total_duration_seconds']:.2f} seconds")
    
    # Option 2: Parallel calls (faster, but more resource-intensive)
    # Uncomment to use parallel calls
    # print("\n\nMethod 2: Parallel calls (max_workers=3)")
    # print("-" * 60)
    # result_parallel = batch_call_phone_numbers(
    #     phone_numbers=phone_numbers,
    #     max_workers=3,  # 3 concurrent calls
    # )
    # 
    # print(f"\nParallel Results:")
    # print(f"  Total calls: {result_parallel['total_calls']}")
    # print(f"  Successful: {result_parallel['successful_calls']}")
    # print(f"  Failed: {result_parallel['failed_calls']}")
    # print(f"  Total duration: {result_parallel['total_duration_seconds']:.2f} seconds")
    
    # Print detailed results
    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print(f"{'='*60}\n")
    
    for result in result_sequential["results"]:
        print(f"Phone: {result['phone_number']}")
        print(f"  Status: {'✓ Success' if result['success'] else '✗ Failed'}")
        if result['success']:
            call_status = result['result']['call_status']
            print(f"  Outcome: {call_status.get('outcome', 'unknown')}")
            print(f"  Duration: {call_status.get('duration_seconds', 0)} seconds")
            print(f"  Call duration: {result['duration_seconds']:.2f} seconds")
            if result['result'].get('transcript'):
                print(f"  Transcript: Available ({len(result['result']['transcript'].get('items', []))} items)")
            else:
                print(f"  Transcript: Not available")
        else:
            print(f"  Error: {result['error']}")
        print()

