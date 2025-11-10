from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

from models import BlogRequest, BlogMeta, BlogState
from llm_providers import load_llm
from graph import build_graph


def generate_blog(req: BlogRequest, provider: str | None = None) -> Dict[str, Any]:
    """Generate a single blog post and return {meta, markdown}."""
    load_dotenv()
    llm = load_llm(provider or os.getenv("AI_PROVIDER", "gemini"))
    app = build_graph(llm)
    init_state: BlogState = {"request": req}
    final_state: BlogState = app.invoke(init_state)

    meta = final_state["meta"]
    markdown = final_state.get("refined_markdown") or final_state.get("draft_markdown")
    return {"meta": meta.dict(), "markdown": markdown}


def write_markdown(meta: BlogMeta, markdown: str, out_dir: str = "./blog_output") -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = meta.slug
    path = os.path.join(out_dir, f"{slug}.md")
    # Escape quotes in summary for YAML frontmatter
    escaped_summary = meta.summary.replace('"', '\\"')
    fm = (
        "---\n"
        f'title: "{meta.title}"\n'
        f'summary: "{escaped_summary}"\n'
        f"keywords: {json.dumps(meta.keywords)}\n"
        f'created_at: "{datetime.now().isoformat()}"\n'
        f"reading_time_min: {meta.reading_time_min}\n"
        "status: draft\n"
        "---\n\n"
    )
    
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(fm + markdown.strip() + "\n")
    return path


def generate_batch(requests: List[BlogRequest], provider: str | None = None, out_dir: str | None = None):
    results = []
    for req in requests:
        out = generate_blog(req, provider)
        meta = BlogMeta(**out["meta"])  # type: ignore[arg-type]
        md = out["markdown"]
        file_path = None
        if out_dir:
            file_path = write_markdown(meta, md, out_dir)
        results.append({"meta": meta.dict(), "markdown": md, "file": file_path})
    return results
