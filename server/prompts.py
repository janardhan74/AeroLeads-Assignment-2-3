from langchain_core.prompts import ChatPromptTemplate

SYSTEM_BASE = (
    "You are an expert technical writer for software engineers. "
    "Write accurate, non-hallucinatory content. Prefer general guidance when unsure of specific versions. "
    "Use GitHub-flavored Markdown."
)

OUTLINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + " Return STRICT JSON for the outline."),
    (
        "human",
        """
Create a detailed outline for a blog.

Inputs:
- Topic: {topic}
- Audience: {audience}
- Tone: {tone}
- Style: {style}
- Target length (words): {length}
- Keywords (comma-separated, optional): {keywords}
- Include code samples: {include_code}

Return STRICT JSON with keys:
{{
  "title": "Concise SEO-friendly title",
  "summary": "1-2 sentence abstract",
  "keywords": ["list","of","keywords"],
  "sections": [
    {{"h2": "Section name", "bullets": ["point 1","point 2"], "code_idea": "optional"}},
    {{"h2": "..."}}
  ]
}}
If unsure about facts, keep them generic. No extra commentary — JSON ONLY.
        """,
    ),
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    (
        "human",
        """
Write a full blog post in Markdown using this outline JSON and requirements.

Outline JSON:
{outline_json}

Requirements:
- H1 at top matching the title.
- Add a concise TL;DR if include_tldr is true.
- Use H2 for sections from outline.
- Use H3 for sub-steps where helpful.
- Prefer concrete examples. If include_code is true, include at least one code block.
- Keep the tone: {tone}; style: {style}; audience: {audience}.
- Target ~{length} words (±15%).
- Avoid making up specific version numbers unless common knowledge.
- Finish with a "Further Reading" list (3–5 reputable links) if include_further_reading is true.

Return ONLY the Markdown, no extra commentary.
        """,
    ),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    (
        "human",
        """
Refine this Markdown blog post to improve clarity, flow, and developer usability.
- Ensure terminology consistency and remove repetition.
- Keep code blocks clean and runnable (language fences where possible).
- Preserve headings and structure; do not shorten below ~{min_words} words.
- Ensure the TL;DR is present if requested.
- Keep facts generic if unsure.

Original Markdown:

{draft_markdown}

Return ONLY the improved Markdown.
        """,
    ),
])
