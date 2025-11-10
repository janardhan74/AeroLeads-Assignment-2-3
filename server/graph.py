from __future__ import annotations
import json
import re
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser

from models import BlogState, BlogMeta
from prompts import OUTLINE_PROMPT, DRAFT_PROMPT, REFINE_PROMPT

slugify_pattern = re.compile(r"[^a-z0-9-]+")

def slugify(title: str) -> str:
    s = title.lower().strip().replace(" ", "-")
    s = slugify_pattern.sub("-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def outline_node(state: BlogState, llm) -> BlogState:
    req = state["request"]
    chain = OUTLINE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "topic": req.topic,
        "audience": req.audience,
        "tone": req.tone,
        "style": req.style,
        "length": req.length,
        "keywords": req.keywords or "",
        "include_code": req.include_code,
    })
    try:
        data: Dict[str, Any] = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group(0))
        else:
            raise RuntimeError(f"Outline JSON parse failed: {raw[:300]}...")

    title = data.get("title") or req.topic
    slug = slugify(title)
    summary = data.get("summary") or f"Overview of {req.topic}."
    keywords = data.get("keywords") or []
    meta = BlogMeta(title=title, slug=slug, summary=summary, keywords=keywords, reading_time_min=max(5, req.length // 200))

    return {**state, "outline_json": data, "meta": meta}


def draft_node(state: BlogState, llm) -> BlogState:
    req = state["request"]
    outline_json = state["outline_json"]
    chain = DRAFT_PROMPT | llm | StrOutputParser()
    md = chain.invoke({
        "outline_json": json.dumps(outline_json, ensure_ascii=False, indent=2),
        "tone": req.tone,
        "style": req.style,
        "audience": req.audience,
        "length": req.length,
        "include_further_reading": req.include_further_reading,
        "include_tldr": req.include_tldr,
    })
    return {**state, "draft_markdown": md}


def refine_node(state: BlogState, llm) -> BlogState:
    req = state["request"]
    draft = state["draft_markdown"]
    chain = REFINE_PROMPT | llm | StrOutputParser()
    refined = chain.invoke({
        "draft_markdown": draft,
        "min_words": int(req.length * 0.85),
    })
    return {**state, "refined_markdown": refined}


def build_graph(llm):
    graph = StateGraph(BlogState)

    def _outline(s: BlogState):
        return outline_node(s, llm)

    def _draft(s: BlogState):
        return draft_node(s, llm)

    def _refine(s: BlogState):
        return refine_node(s, llm)

    graph.add_node("outline", _outline)
    graph.add_node("draft", _draft)
    graph.add_node("refine", _refine)

    graph.set_entry_point("outline")
    graph.add_edge("outline", "draft")
    graph.add_edge("draft", "refine")
    graph.add_edge("refine", END)

    return graph.compile()
