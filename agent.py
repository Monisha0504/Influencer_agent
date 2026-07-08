"""
Influencer Discovery Agent - LangChain core logic.

Flow:
    scrape (optional) -> build prompt -> LLM via OpenRouter -> parse JSON
                      -> overwrite profile_link with Tavily real URL
                      -> remember names so the NEXT run for the same inputs
                         returns DIFFERENT creators
"""

import json
import os
import re
import urllib.parse
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------- Config ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", os.getenv("MODEL_NAME", "openai/gpt-4o-mini")).strip()
APP_NAME           = os.getenv("APP_NAME", "Influencer Discovery Agent")
APP_URL            = os.getenv("APP_URL", "http://localhost:8000")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "").strip()

# Total influencers we ask for per run
TARGET_COUNT = 50
# How many recent names to remember per (category, followers_range)
MEMORY_SIZE = 200

if not OPENROUTER_API_KEY:
    print("[WARN] OPENROUTER_API_KEY is not set. Put it in .env before running.")


# ---------- "Don't repeat" memory ----------
# Maps (category_lc, followers_range_lc) -> deque of recently returned names.
# Lives only in this process - cleared on restart.
_seen_creators: Dict[Tuple[str, str], Deque[str]] = defaultdict(
    lambda: deque(maxlen=MEMORY_SIZE)
)


def _memory_key(category: str, followers_range: str) -> Tuple[str, str]:
    return (category.strip().lower(), followers_range.strip().lower())


def _remember_names(category: str, followers_range: str, names: List[str]) -> None:
    bucket = _seen_creators[_memory_key(category, followers_range)]
    for n in names:
        if n and n not in bucket:
            bucket.append(n)


def _names_to_avoid(category: str, followers_range: str) -> List[str]:
    return list(_seen_creators[_memory_key(category, followers_range)])


# ---------- Website scraping ----------
def scrape_website(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return cleaned plain text (best-effort)."""
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[Could not fetch website: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# ---------- Profile link builder (Tavily-powered) ----------
_profile_cache: dict = {}


def build_profile_url(name: str, platform: str) -> str:
    """Use Tavily to find the real profile URL. Falls back to search URL."""
    if not name:
        return ""

    cache_key = f"{name.lower()}::{platform.lower()}"
    if cache_key in _profile_cache:
        return _profile_cache[cache_key]

    platform_lc = (platform or "").lower().strip()

    domain_map = {
        "instagram": "instagram.com",
        "youtube":   "youtube.com",
        "facebook":  "facebook.com",
        "linkedin":  "linkedin.com",
        "tiktok":    "tiktok.com",
        "x":         "x.com",
        "twitter":   "x.com",
    }
    domain = domain_map.get(platform_lc, "")

    # Try Tavily to get the real profile URL
    if TAVILY_API_KEY and domain:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=f"{name} {platform} official profile",
                search_depth="basic",
                max_results=5,
                include_domains=[domain],
            )
            for result in response.get("results", []):
                url = result.get("url", "")
                if domain in url and _is_profile_url(url, platform_lc):
                    _profile_cache[cache_key] = url
                    print(f"[Tavily HIT] {name} / {platform} → {url}")
                    return url
        except Exception as e:
            print(f"[Tavily] Failed for {name} / {platform}: {e}")
        print(f"[Tavily MISS] {name} / {platform} — using fallback")

    # Fallback: search URLs
    url = _fallback_search_url(name, platform_lc)
    _profile_cache[cache_key] = url
    return url


def _is_profile_url(url: str, platform: str) -> bool:
    """Return True if the URL looks like a profile page, not a post/video."""
    bad = ["/p/", "/reel/", "/watch?", "/shorts/", "/posts/", "/photo/", "/status/"]
    if any(b in url for b in bad):
        return False
    if platform == "instagram":
        parts = [p for p in url.rstrip("/").split("/") if p and "instagram.com" not in p]
        return len(parts) == 1
    if platform == "youtube":
        return any(x in url for x in ["/@", "/channel/", "/c/", "/user/"])
    if platform in ("x", "twitter"):
        parts = [p for p in url.rstrip("/").split("/") if p and "x.com" not in p]
        return len(parts) == 1
    if platform == "linkedin":
        return "/in/" in url or "/company/" in url
    if platform == "facebook":
        return "/people/" in url or (
            url.count("/") <= 4 and "/pg/" not in url and "/groups/" not in url
        )

    return True


def _fallback_search_url(name: str, platform_lc: str) -> str:
    """Search-page URLs as a last resort."""
    q = urllib.parse.quote_plus(name)
    if "youtube"   in platform_lc: return f"https://www.youtube.com/results?search_query={q}"
    if "instagram" in platform_lc: return f"https://www.google.com/search?q=site%3Ainstagram.com+{q}"
    if "facebook" in platform_lc: return f"https://www.facebook.com/search/people/?q={q}"
    if "linkedin"  in platform_lc: return f"https://www.linkedin.com/search/results/people/?keywords={q}"
    if "tiktok"    in platform_lc: return f"https://www.tiktok.com/search/user?q={q}"
    if platform_lc in ("x", "twitter", "x/twitter", "x (twitter)"): return f"https://x.com/search?q={q}&f=user"
    return f"https://www.google.com/search?q={q}+{urllib.parse.quote_plus(platform_lc)}"


# ---------- LLM (OpenRouter via LangChain) ----------
def get_llm() -> ChatOpenAI:
    """OpenRouter is OpenAI-compatible. We point ChatOpenAI at it."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0.85,
        max_tokens=12000,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": APP_NAME,
        },
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ---------- Prompt ----------
SYSTEM_PROMPT = """You are an expert Influencer Marketing Strategist who specialises in the global creator ecosystem.
Return ONLY a single valid JSON object. No markdown fences. No commentary."""

USER_TEMPLATE = """Recommend influencers for this brief.

Industry / Creator Category: {category}
Required follower range: {followers_range}
Location Filter: {location_label}
Platform Filter: {platforms_label}
Website (optional): {website}

Scraped website content (may be empty):
{content}

============================================================
HARD REQUIREMENTS - every one is mandatory:
============================================================

1. QUANTITY: Return EXACTLY {target_count} influencers. The "influencers"
   array MUST have length {target_count}. Never drop below {target_count}.

{platform_block}

{location_block}

4. FOLLOWERS: Every influencer's "followers" count MUST fall inside the
   requested range: {followers_range}.

5. FIT SCORE: integer 0-100.

6. CONTENT STRATEGY: one concrete actionable idea
   (Reel, Tutorial, Case study, Product demo, Unboxing, Carousel, Live AMA, etc.).

7. PROFILE LINK: include a plausible URL - we will post-process server-side.

8. NO REPEATS: {avoid_block}

{preference_match_requirement}

============================================================
OUTPUT JSON SHAPE - exactly these keys, no extras, no omissions:
============================================================

{{
  "category": "",
  "followers_range": "",
  "website": "",
  "industry": "",
  "target_audience": "",
  "recommended_platforms": [],
  "influencers": [
    {{
      "name": "",
      "platform": "",
      "niche": "",
      "location": "",
      "followers": "",
      "engagement_rate": "",
      "profile_link": "",
      "fit_score": 0,
      "fit_reason": "",
      "competitor_collaboration": "",
      "content_strategy": "",
      "preference_match": ""
    }}
  ]
}}

REMEMBER: {target_count} influencers. Respect platform and location filters strictly. JSON only.
"""

VALID_PLATFORMS = ["Instagram", "YouTube", "Facebook", "LinkedIn", "X", "TikTok"]


def _build_platform_block(platforms: Optional[List[str]], target_count: int) -> str:
    selected = [p for p in (platforms or []) if p in VALID_PLATFORMS]

    if not selected:
        return (
            f"2. PLATFORM DIVERSITY: The {target_count} picks MUST cover multiple platforms.\n"
            f"   MINIMUMS for a list of {target_count}:\n"
            f"       - At least 12 on Instagram\n"
            f"       - At least 10 on YouTube\n"
            f"       - At least  7 on Facebook\n"
            f"       - At least  5 on LinkedIn\n"
            f"       - At least  3 on X (Twitter)\n"
            f"   Distribute the remaining slots wherever fits best.\n"
            f'   Acceptable "platform" values: "Instagram", "YouTube", "Facebook", "LinkedIn", "X".'
        )

    platform_list = ", ".join(f'"{p}"' for p in selected)

    if len(selected) == 1:
        return (
            f"2. PLATFORM: ALL {target_count} picks MUST be on {selected[0]} ONLY.\n"
            f"   Do NOT include influencers from any other platform.\n"
            f'   The only acceptable "platform" value is: {platform_list}.'
        )

    base = target_count // len(selected)
    remainder = target_count - base * len(selected)
    lines = [
        f"2. PLATFORM RESTRICTION: ONLY include influencers from these selected platforms: {', '.join(selected)}.",
        f"   Do NOT include any influencer from other platforms.",
        f"   Distribute {target_count} picks across selected platforms with MINIMUMS:",
    ]
    for i, p in enumerate(selected):
        min_count = base + (1 if i < remainder else 0)
        lines.append(f"       - At least {min_count} on {p}")
    lines.append(f'   Only acceptable "platform" values: {platform_list}.')
    return "\n".join(lines)


def _build_location_block(location: Optional[str]) -> str:
    if not location or not location.strip():
        return (
            "3. LOCATION PRIORITY:\n"
            "   - Prefer India-based creators (Bangalore / Karnataka first, then rest of India).\n"
            "   - Only after exhausting strong Indian options should you include global picks.\n"
            '   - Always set "location" to a real city / country (e.g. "Bangalore, India").'
        )
    loc = location.strip()
    return (
        f"3. LOCATION FILTER — THIS IS MANDATORY:\n"
        f'   - You MUST ONLY return influencers who are BASED IN or primarily active in: "{loc}".\n'
        f"   - Influencers from nearby cities within the same region are acceptable if they\n"
        f'     frequently create content for or about "{loc}".\n'
        f"   - Do NOT include influencers from other countries or unrelated distant regions.\n"
        f'   - Every influencer\'s "location" field must reflect a real place near "{loc}".'
    )


def _build_preference_block(preference_directive: Optional[str]) -> str:
    return (preference_directive or "").strip()


def _build_preference_match_requirement(preference_directive: Optional[str]) -> str:
    if not preference_directive:
        return '9. PREFERENCE MATCH: Leave the "preference_match" field as an empty string "".'
    return (
        '9. PREFERENCE MATCH: For EVERY influencer, populate the "preference_match" field with\n'
        "   one or two concrete sentences that PROVE this creator matches the creator preference\n"
        "   stated above. Be specific — cite concrete details (age, background, family members,\n"
        "   content examples) that confirm the match. Do NOT be generic or vague."
    )


def _build_avoid_block(category: str, followers_range: str) -> str:
    avoid = _names_to_avoid(category, followers_range)
    if not avoid:
        return (
            "This is the first run for this brief. Pick the strongest "
            "creators you know of."
        )
    listed = ", ".join(avoid[-MEMORY_SIZE:])
    return (
        "You have already recommended the following creators for this exact "
        "brief in a previous run. Do NOT include any of them again - choose a "
        "completely DIFFERENT roster this time:\n"
        f"{listed}"
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(f"Model did not return JSON. Raw output:\n{text}")
        return json.loads(match.group(0))


def _call_llm(
    category: str,
    followers_range: str,
    website: str,
    content: str,
    location: str = "",
    platforms: Optional[List[str]] = None,
    preference_directive: Optional[str] = None,
    extra: str = "",
) -> dict:
    preference_block = _build_preference_block(preference_directive)
    suffix_parts = []
    if preference_block:
        suffix_parts.append(preference_block)
    if extra:
        suffix_parts.append(f"ADDITIONAL DIRECTIVE:\n{extra}")
    user_template = USER_TEMPLATE + (
        "\n\n" + "\n\n".join(suffix_parts) if suffix_parts else ""
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", user_template)]
    )
    chain = prompt | get_llm() | StrOutputParser()
    raw = chain.invoke(
        {
            "category": category,
            "followers_range": followers_range,
            "location_label": location.strip() if location else "Not specified (prefer India-based creators)",
            "platforms_label": ", ".join(platforms) if platforms else "All platforms",
            "platform_block": _build_platform_block(platforms, TARGET_COUNT),
            "location_block": _build_location_block(location),
            "website": website or "N/A",
            "content": content or "N/A",
            "target_count": TARGET_COUNT,
            "avoid_block": _build_avoid_block(category, followers_range),
            "preference_match_requirement": _build_preference_match_requirement(preference_directive),
        }
    )
    return _extract_json(raw)


# ---------- Public entry point ----------
def run_discovery(
    category: str,
    followers_range: str,
    website: Optional[str] = None,
    location: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    preference_directive: Optional[str] = None,
) -> dict:
    """Run the whole pipeline and return a normalized dict ready for the template."""
    content = scrape_website(website) if website else ""

    parsed = _call_llm(
        category, followers_range, website or "", content,
        location=location or "", platforms=platforms,
        preference_directive=preference_directive,
    )

    # If we got fewer than ~80% of target, retry once with a stronger nudge.
    n = len(parsed.get("influencers") or [])
    if n < int(TARGET_COUNT * 0.8):
        selected_plats = ", ".join(platforms) if platforms else "Instagram, YouTube, Facebook, LinkedIn and X"
        try:
            parsed_2 = _call_llm(
                category,
                followers_range,
                website or "",
                content,
                location=location or "",
                platforms=platforms,
                preference_directive=preference_directive,
                extra=(
                    f"Your previous attempt returned only {n} influencers. "
                    f"You MUST return EXACTLY {TARGET_COUNT}. "
                    f"Ensure {selected_plats} are all represented."
                ),
            )
            if len(parsed_2.get("influencers") or []) > n:
                parsed = parsed_2
        except Exception:
            pass

    # Normalize + overwrite profile_link with Tavily URL
    influencers: List[dict] = []
    for item in parsed.get("influencers", []) or []:
        name     = (item.get("name") or "").strip()
        platform = (item.get("platform") or "").strip()
        link     = build_profile_url(name, platform)
        influencers.append(
            {
                "name":                   name,
                "platform":               platform,
                "niche":                  item.get("niche", "") or "",
                "location":               item.get("location", "") or "",
                "followers":              str(item.get("followers", "") or ""),
                "engagement_rate":        str(item.get("engagement_rate", "") or ""),
                "profile_link":           link,
                "fit_score":              item.get("fit_score", 0) or 0,
                "fit_reason":             item.get("fit_reason", "") or "",
                "competitor_collaboration": item.get("competitor_collaboration", "") or "",
                "content_strategy":       item.get("content_strategy", "") or "",
                "preference_match":       item.get("preference_match", "") or "",
            }
        )

    # Cap to TARGET_COUNT, then commit names to memory so next run avoids them.
    influencers = influencers[:TARGET_COUNT]
    _remember_names(category, followers_range, [i["name"] for i in influencers])

    return {
        "category":             parsed.get("category", category) or category,
        "followers_range":      parsed.get("followers_range", followers_range) or followers_range,
        "website":              parsed.get("website", website or "") or (website or ""),
        "location":             location or "",
        "platforms":            platforms or [],
        "preference_directive":  preference_directive or "",
        "industry":             parsed.get("industry", "") or "",
        "target_audience":      parsed.get("target_audience", "") or "",
        "recommended_platforms": parsed.get("recommended_platforms", []) or [],
        "influencers":          influencers,
    }