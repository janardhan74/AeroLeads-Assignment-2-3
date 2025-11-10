# AeroLeads App

A comprehensive full-stack application featuring AI-powered blog generation and intelligent voice agent capabilities for making phone calls.

## 🚀 Features

### 📝 AI Blog Generator

- Generate high-quality blog posts using AI (Gemini, OpenAI, or Perplexity)
- Customizable blog parameters (topic, audience, tone, style, length)
- Support for code examples, TLDR summaries, and further reading sections
- Markdown-based blog storage and viewing

### 📞 Voice Agent

- **Single Call**: Make individual phone calls with full transcript support
- **Agent Call**: Natural language interface to make calls (e.g., "make call to +1234567890")
- **Multiple Calls**: Batch calling with file upload support
- Real-time call status monitoring
- Automatic transcript generation and storage
- Support for multiple phone numbers in a single prompt

## 🏗️ Architecture

```
AeroLeads App/
├── client/
│   └── frontend/          # React + TypeScript frontend
└── server/                 # FastAPI backend
    └── voice_agent/       # Voice agent implementation
```

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+
- **LiveKit** account and credentials
- **AI Provider API Keys** (Gemini, OpenAI, or Perplexity)
- **Twilio** account (for SIP trunk configuration)

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd App
```

### 2. Backend Setup

```bash
cd server
python -m venv cvenv
source cvenv/bin/activate  # On Windows: cvenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd client/frontend
npm install
```

### 4. Environment Variables

Create a `.env` file in the `server` directory:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
OUTBOUND_TRUNK_ID=your-trunk-id
FROM_NUMBER=+1234567890
AGENT_NAME=aeroleads-voice-agent

# AI Provider (choose one)
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
PERPLEXITY_API_KEY=your-perplexity-key
AI_PROVIDER=gemini  # or openai, perplexity

# Blog Configuration
BLOG_DIR=./blog_output
TRANSCRIPTS_DIR=./transcripts
```

### 5. Run the Application

**Backend:**

```bash
cd server
python run.py
# or
uvicorn main:app --reload --port 8000
```

**Frontend:**

```bash
cd client/frontend
npm run dev
```

The application will be available at:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📚 Documentation

- [Frontend Documentation](./client/frontend/README.md)
- [Backend Documentation](./server/README.md)

## 🔧 Technology Stack

### Backend

- **FastAPI** - Modern Python web framework
- **LiveKit** - Real-time voice communication
- **LangChain** - LLM orchestration
- **LangGraph** - Agent workflow management
- **Google Gemini / OpenAI / Perplexity** - AI providers

### Frontend

- **React 19** - UI library
- **TypeScript** - Type safety
- **Material-UI (MUI) v7** - Component library
- **Vite** - Build tool
- **React Router** - Routing

## 📖 API Endpoints

### Blog Generation

- `POST /api/generate-articles` - Generate blog posts
- `GET /api/blogs` - List all blogs
- `GET /api/blogs/{slug}` - Get blog by slug

### Voice Agent

- `POST /api/voice/make-a-call` - Make a single call
- `POST /api/voice/make-agent-call` - Agent-based call with natural language
- `POST /api/voice/make-multiple-calls` - Batch calling

See [Backend README](./server/README.md) for detailed API documentation.

## 🎯 Usage Examples

### Generate a Blog

1. Navigate to "Generate Blog" in the frontend
2. Enter your topic and customize parameters
3. Click "Generate Blog"
4. View the generated blog post

### Make a Voice Call

1. Navigate to "Voice Agent"
2. Choose a tab:
   - **Make Call**: Direct phone call
   - **Agent Call**: Natural language (e.g., "make call to +1234567890 and +9876543210")
   - **Multiple Calls**: Upload a file with phone numbers or enter manually

## 📁 Project Structure

```
App/
├── client/
│   └── frontend/
│       ├── src/
│       │   ├── components/     # Reusable components
│       │   ├── pages/          # Page components
│       │   └── App.tsx         # Main app component
│       └── package.json
├── server/
│   ├── voice_agent/           # Voice agent modules
│   ├── blog_routes.py         # Blog API routes
│   ├── voice_agent_routes.py  # Voice agent API routes
│   ├── main.py                # FastAPI app
│   └── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

[Add your license here]

## 🆘 Support

For issues and questions, please open an issue on the repository.

## 🔐 Security Notes

- Never commit `.env` files
- Keep API keys secure
- Use environment variables for sensitive data
- Review CORS settings for production
