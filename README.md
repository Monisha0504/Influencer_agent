# Influencer Discovery Agent (LangChain + FastAPI + OpenRouter)

A Python port of your n8n "Influencer Discovery Agent" workflow.
You fill in a small form (category, follower range, optional website) and the agent returns 10 ranked influencer recommendations as a styled HTML page.

## What it does

1. Form collects **category**, **follower range** (preset or manual), and an **optional website**.
2. If a website is given, the homepage is scraped and cleaned to ~5000 chars.
3. A LangChain prompt is sent to **OpenRouter** (default model: `openai/gpt-4o-mini`).
4. The JSON response is parsed and rendered as a results page with fit-score badges, platform chips, profile links, and content-strategy ideas.

## Project structure

```
influencer_agent/
├── main.py              # FastAPI app
├── agent.py             # LangChain logic (scrape + LLM + parse)
├── requirements.txt
├── .env.example
├── templates/
│   ├── index.html       # Input form
│   └── results.html     # Results page
└── static/
    └── style.css
```

## Setup

1. **Clone / open the folder in VS Code.**

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenRouter:**
   - Copy `.env.example` to `.env`
   - Paste your key into `OPENROUTER_API_KEY`
   - (Optional) change `OPENROUTER_MODEL` — any OpenRouter model id works, e.g.
     `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `google/gemini-flash-1.5`.

   Get a key at <https://openrouter.ai/keys>.

## Run

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>.

## Inputs

| Field | Required | Notes |
|---|---|---|
| Industry / Creator Category | Yes | Dropdown of 20 common niches (Finance, Food Blog, Marketing, Tech, Fashion, Fitness, etc.). A free-text override field lets you type any custom category. |
| Follower Range | Yes | Toggle between a preset dropdown (Nano / Micro / Mid / Macro / Mega) and a manual `min — max` input. |
| Website | No | If provided, the homepage is scraped to ground the recommendations. |

## How OpenRouter is wired up

OpenRouter is OpenAI-compatible, so we use `ChatOpenAI` from `langchain-openai` and override the base URL:

```python
ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-Title": APP_NAME,
    },
)
```

You can swap models freely by changing `OPENROUTER_MODEL` in `.env`.

## Troubleshooting

- **"Model did not return JSON"** — Some smaller models occasionally wrap output in prose. We strip code fences and regex-extract JSON, but if you see this, switch to a stronger model (e.g. `anthropic/claude-3.5-sonnet`).
- **Website scrape fails** — Many sites block default user agents. The agent will continue without scraped content; recommendations rely on the category instead.
- **CORS / port conflict** — Change the port: `uvicorn main:app --reload --port 8080`.
