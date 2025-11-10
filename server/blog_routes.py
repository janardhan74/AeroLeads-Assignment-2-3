# blog_routes.py
from __future__ import annotations

import os
import time
import zipfile
import re
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from models import BlogRequest
from generate import generate_batch

load_dotenv()

BLOG_DIR = os.getenv("BLOG_DIR", "./blog_output")
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "gemini")

# Create router for blog routes
router = APIRouter(prefix="/api", tags=["blogs"])


class GeneratePayload(BaseModel):
    provider: Optional[str] = None  # "gemini" | "openai" | "perplexity"
    items: List[BlogRequest]        # Only 'topic' must be present if you made others optional


def _ensure_blog_dir() -> None:
    os.makedirs(BLOG_DIR, exist_ok=True)


def _slug_to_path(slug: str) -> str:
    # All posts are written as BLOG_DIR/<slug>.md by generate_batch()
    return os.path.join(BLOG_DIR, f"{slug}.md")


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from markdown file."""
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    if match:
        frontmatter_text = match.group(1)
        metadata = {}
        for line in frontmatter_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                metadata[key] = value
        return metadata
    return {}


@router.post("/generate-articles")
async def generate_articles(
    payload: GeneratePayload,
    as_zip: bool = Query(False, description="If true, return a ZIP file containing all generated posts"),
):
    """
    Generate articles.

    - If `as_zip=false` (default): returns JSON with metadata and per-file **download URLs**.
      The file content is NOT embedded in JSON.
    - If `as_zip=true`: returns a **FileResponse** with a ZIP attachment of all generated .md files.
    """
    try:
        _ensure_blog_dir()

        results = generate_batch(
            requests=payload.items,
            provider=payload.provider or DEFAULT_PROVIDER,
            out_dir=BLOG_DIR,
        )
        # results: [{"meta": {...}, "markdown": "...", "file": "/abs/or/rel/path.md"}, ...]

        if as_zip:
            # Bundle all generated files into a ZIP and return it as a file attachment
            ts = int(time.time())
            zip_name = f"generated-articles-{ts}.zip"
            zip_path = os.path.join(BLOG_DIR, zip_name)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    meta = r["meta"]
                    slug = meta["slug"]
                    file_path = r["file"] or _slug_to_path(slug)
                    if not os.path.isfile(file_path):
                        # If for some reason file wasn't written, skip it gracefully
                        continue
                    # Store inside the ZIP as "<slug>.md"
                    arcname = os.path.basename(file_path)
                    zf.write(file_path, arcname=arcname)

            # Return the ZIP as an attachment (no JSON body)
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename=zip_name,
            )

        # Default: JSON with metadata + download URLs (not the content)
        created = []
        for r in results:
            meta = r["meta"]
            slug = meta["slug"]
            # We serve files via /api/file/{slug}
            created.append(
                {
                    "meta": meta,
                    "download_url": f"/api/file/{slug}",
                    "file": r["file"],  # filesystem path for debugging/logging if you need it
                }
            )

        return JSONResponse(
            content={
                "created": created,
                "count": len(created),
                "dir": BLOG_DIR,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blogs")
async def list_blogs():
    """
    List all available blog files with their metadata.
    """
    try:
        _ensure_blog_dir()
        blog_files = []
        
        for file_path in Path(BLOG_DIR).glob("*.md"):
            slug = file_path.stem
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                metadata = _parse_frontmatter(content)
                metadata["slug"] = slug
                metadata["filename"] = file_path.name
                
                blog_files.append(metadata)
            except Exception as e:
                # Skip files that can't be read
                continue
        
        return JSONResponse(content={"blogs": blog_files, "count": len(blog_files)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blog/{slug}")
async def get_blog_content(slug: str):
    """
    Get the full content of a blog post by slug.
    """
    path = _slug_to_path(slug)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse frontmatter
        metadata = _parse_frontmatter(content)
        
        # Remove frontmatter from content
        content_without_frontmatter = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        
        return JSONResponse(content={
            "slug": slug,
            "metadata": metadata,
            "content": content_without_frontmatter.strip()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file/{slug}")
async def download_file(slug: str):
    """
    Download a single generated markdown file by slug, as a raw file.
    This preserves the exact bytes/formatting (no JSON wrapping).
    """
    path = _slug_to_path(slug)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    # Serve with a markdown content type; most browsers will download or display raw.
    return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=f"{slug}.md")


