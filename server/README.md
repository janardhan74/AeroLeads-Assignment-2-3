# Backend - AeroLeads App

FastAPI-based backend server providing AI blog generation and voice agent capabilities.

## 🚀 Tech Stack

- **FastAPI** - Modern Python web framework
- **LiveKit** - Real-time voice communication and SIP integration
- **LangChain** - LLM orchestration framework
- **LangGraph** - Agent workflow management
- **Google Gemini / OpenAI / Perplexity** - AI providers
- **Uvicorn** - ASGI server

## 📋 Prerequisites

- Python 3.11+
- LiveKit account and credentials
- AI Provider API key (Gemini, OpenAI, or Perplexity)
- Twilio account (for SIP trunk)

## 🛠️ Installation

### 1. Create Virtual Environment

```bash
python -m venv cvenv

# Activate virtual environment
# On Windows:
cvenv\Scripts\activate
# On macOS/Linux:
source cvenv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `server` directory:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
OUTBOUND_TRUNK_ID=your-trunk-id
FROM_NUMBER=+1234567890
AGENT_NAME=aeroleads-voice-agent
SIP_PARTICIPANT_ID=pstn-callee

# AI Provider Configuration
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-001
OPENAI_API_KEY=your-openai-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key
AI_PROVIDER=gemini  # Options: gemini, openai, perplexity

# Blog Configuration
BLOG_DIR=./blog_output
TRANSCRIPTS_DIR=./transcripts
```

## 🏃 Running the Server

### Development Mode

```bash
# Using run.py
python run.py

# Or using uvicorn directly
uvicorn main:app --reload --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## 📁 Project Structure

```
server/
├── voice_agent/
│   ├── agent_call_tool.py          # LangGraph agent with call tools
│   ├── make_call_with_transcript.py # Core call functionality
│   ├── batch_call.py                # Batch calling implementation
│   ├── agent_with_transcripts.py    # Agent with transcript support
│   └── ...
├── blog_routes.py                   # Blog generation API routes
├── voice_agent_routes.py            # Voice agent API routes
├── main.py                          # FastAPI application
├── models.py                        # Pydantic models
├── llm_providers.py                 # LLM provider abstraction
├── generate.py                      # Blog generation logic
├── graph.py                         # LangGraph blog generation graph
├── prompts.py                       # Prompt templates
├── run.py                           # Server runner script
└── requirements.txt                 # Python dependencies
```

## 📡 API Endpoints

### Health Check

**GET** `/health`

- Returns server status and configuration

### Blog Generation

**POST** `/api/generate-articles`

- Generate blog posts using AI
- Request body:
  ```json
  {
    "provider": "gemini",
    "items": [
      {
        "topic": "Getting Started with React",
        "audience": "Developers",
        "tone": "Clear, friendly",
        "style": "How-to tutorial",
        "length": 1400,
        "keywords": "React, Tutorial",
        "include_code": true,
        "include_tldr": true,
        "include_further_reading": true
      }
    ]
  }
  ```

**GET** `/api/blogs`

- List all generated blogs

**GET** `/api/blogs/{slug}`

- Get blog by slug

### Voice Agent

**POST** `/api/voice/make-a-call`

- Make a single phone call
- Request body:
  ```json
  {
    "phone_number": "+1234567890",
    "room_name": "optional-room-name",
    "from_number": "+0987654321",
    "wait_until_answered": true,
    "transcript_timeout": 30.0
  }
  ```
- Response includes call status and transcript

**POST** `/api/voice/make-agent-call`

- Agent-based call with natural language
- Request body:
  ```json
  {
    "prompt": "make call to +1234567890 and +9876543210"
  }
  ```
- Supports single or multiple phone numbers
- Response includes agent response and call results

**POST** `/api/voice/make-multiple-calls`

- Batch calling with multiple phone numbers
- Request body:
  ```json
  {
    "phone_numbers": ["+1234567890", "+9876543210"],
    "room_name_prefix": "batch-call",
    "from_number": "+0987654321",
    "wait_until_answered": true,
    "transcript_timeout": 30.0,
    "delay_between_calls": 2.0
  }
  ```
- Response includes batch statistics and individual call results

## 🔧 Voice Agent Features

### Single Call

- Direct phone call with full control
- Real-time call status monitoring
- Automatic transcript generation
- Call duration and outcome tracking

### Agent Call

- Natural language interface
- Automatic phone number extraction
- Support for multiple numbers in one prompt
- Intelligent call routing
- Comprehensive response with all call details

### Multiple Calls

- Batch processing
- Sequential or configurable delay between calls
- Individual call tracking
- Aggregate statistics
- Error handling per call

## 📝 Voice Agent Architecture

### Agent Implementation

- **LangGraph** for agent workflow
- **LangChain** for LLM integration
- **Tool-based** architecture for call placement
- Automatic tool selection based on prompt

### Call Flow

1. User provides prompt (single or multiple numbers)
2. Agent analyzes prompt and selects appropriate tool
3. Phone numbers are extracted and normalized
4. Calls are made sequentially
5. Transcripts are collected after each call
6. Results are aggregated and returned

### Transcript Management

- Automatic transcript generation
- File-based storage in `transcripts/` directory
- JSON format with conversation items
- Support for multiple transcript formats

## 🔐 Security

- Environment variables for sensitive data
- CORS middleware for frontend integration
- API key validation
- Input sanitization

## 🧪 Testing

### Test Voice Agent

```bash
# Test agent call
python voice_agent/test_run_call.py

# Test batch calls
python voice_agent/example_batch_call.py
```

### Test Blog Generation

```bash
# Use the CLI
python cli.py generate --topic "Your Topic"
```

## 📊 Monitoring

- Health check endpoint for monitoring
- Call status tracking
- Transcript logging
- Error handling and reporting

## 🐛 Troubleshooting

### LiveKit Connection Issues

- Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`
- Check network connectivity
- Ensure LiveKit server is accessible

### Call Failures

- Verify `OUTBOUND_TRUNK_ID` is correct
- Check `FROM_NUMBER` format (E.164)
- Ensure SIP trunk is properly configured in LiveKit

### Transcript Not Generated

- Check `TRANSCRIPTS_DIR` exists and is writable
- Verify agent is running and processing calls
- Check `transcript_timeout` value

### AI Provider Issues

- Verify API keys are correct
- Check API rate limits
- Ensure provider service is accessible

## 🚀 Production Deployment

### Recommended Setup

1. Use a production ASGI server (Gunicorn with Uvicorn workers)
2. Set up reverse proxy (nginx)
3. Use environment variables for all configuration
4. Enable HTTPS
5. Set up proper logging
6. Configure CORS for production domain

### Example Production Command

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LiveKit Documentation](https://docs.livekit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 🔄 Dependencies

Key dependencies:

- `fastapi` - Web framework
- `livekit` - Voice communication
- `livekit-agents` - Agent framework
- `langchain` - LLM orchestration
- `langchain-google-genai` - Gemini integration
- `openai` - OpenAI integration
- `python-dotenv` - Environment management
- `uvicorn` - ASGI server

See `requirements.txt` for complete list.
