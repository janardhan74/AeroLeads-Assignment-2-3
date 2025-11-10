# backend/fastapi_app/main.py
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import route routers
from blog_routes import router as blog_router
from voice_agent_routes import router as voice_agent_router

load_dotenv()

BLOG_DIR = os.getenv("BLOG_DIR", "./blog_output")
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "gemini")

app = FastAPI(title="AI Blog Generator & Voice Agent API", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(blog_router)
app.include_router(voice_agent_router)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "provider": DEFAULT_PROVIDER,
        "dir": BLOG_DIR,
    }
