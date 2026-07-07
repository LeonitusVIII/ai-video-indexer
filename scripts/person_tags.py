"""Extract person-related tags from vision and transcript text (no face-recognition model)."""
import re

PERSON_PATTERNS = [
    re.compile(r"\b(?:a|an|the)\s+(man|woman|boy|girl|child|baby|toddler|teen(?:ager)?|adult|elderly\s+man|elderly\s+woman|person|people|group\s+of\s+people)\b", re.I),
    re.compile(r"\b(mother|father|mom|dad|grandma|grandpa|grandmother|grandfather|son|daughter|brother|sister|family)\b", re.I),
    re.compile(r"\b(man|woman|boy|girl|child|baby|person)\s+(?:in|with|wearing|holding|standing|sitting|walking|running)\b", re.I),
]

NAME_PATTERN = re.compile(
    r"\b(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


def _normalize_tag(text):
    text = " ".join(text.split())
    if len(text) < 3 or len(text) > 80:
        return None
    return text.strip().lower()


def extract_people_tags(vision_data=None, transcript_data=None, *, max_tags=20):
    """Return deduplicated person-related tags from vision frame descriptions."""
    texts = []
    for frame in (vision_data or {}).get("frames") or []:
        desc = (frame.get("description") or "").strip()
        if desc:
            texts.append(desc)

    for segment in (transcript_data or {}).get("segments") or []:
        text = (segment.get("text") or "").strip()
        if text:
            texts.append(text)

    tags = []
    seen = set()
    for text in texts:
        for pattern in PERSON_PATTERNS:
            for match in pattern.finditer(text):
                tag = _normalize_tag(match.group(0))
                if tag and tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        for match in NAME_PATTERN.finditer(text):
            tag = _normalize_tag(f"named {match.group(1)}")
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)

    return tags[:max_tags]
