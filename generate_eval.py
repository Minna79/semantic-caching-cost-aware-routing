import os, json, csv, random, requests
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()
GEN_MODEL = "anthropic/claude-sonnet-5.5"   # strong model writes the questions
DOCS_DIR = Path("docs")
QUESTIONS_PER_DOC = 3
MAX_CHARS = 6000                            # text sent per document

def read_doc(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return ""

PROMPT = """Below is a document. Write {n} question-answer pairs a real user might ask about it.

Rules:
- Each answer must be fully supported by the document text, and short (1-2 sentences).
- Make the questions specific, not vague like "what is this document about".
- Mix difficulty: at least one simple fact lookup and, if possible, one that needs
  combining two facts or explaining something.
- Write the questions and answers in the SAME language as the document.
- Return ONLY a JSON list, no markdown: [{{"question": "...", "answer": "...", "difficulty": "simple" or "complex"}}]

Document:
{text}"""



def ask(prompt: str, retries=5) -> str:
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEN_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,   # 3 short QA pairs need far less than 2000
                },
                timeout=120,
            )
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  network error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
            continue

        data = r.json()
        if r.status_code in (402, 429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
            wait = min(wait, 150)
            print(f"  HTTP {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200 or "error" in data:
            raise RuntimeError(f"HTTP {r.status_code}: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    raise RuntimeError("gave up after retries")


def parse_json_list(raw: str):
    """Parse the model's reply, tolerating ```json fences or stray text."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON list found")
    return json.loads(raw[start:end + 1])

rows = []
for path in sorted(DOCS_DIR.iterdir()):
    text = read_doc(path).strip()
    if len(text) < 500:
        print(f"skip {path.name} (empty or too short)")
        continue

    lang = path.stem.split("_")[0]  # en_00.txt -> "en"
    prompt = PROMPT.format(n=QUESTIONS_PER_DOC, text=text[:MAX_CHARS])

    try:
        items = parse_json_list(ask(prompt))
    except Exception as e:
        print(f"FAIL {path.name}: {e}")
        continue

    kept = 0
    for it in items:
        q, a = it.get("question"), it.get("answer")
        if not q or not a:
            continue
        rows.append({
            "doc": path.name,
            "lang": lang,
            "question": q.strip(),
            "answer": a.strip(),
            "difficulty": it.get("difficulty", ""),
        })
        kept += 1
    print(f"ok   {path.name}: {kept} questions")

if not rows:
    raise SystemExit("No questions generated. Check the errors above.")

random.seed(0)
random.shuffle(rows)
with open("eval_set_raw.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {len(rows)} questions to eval_set_raw.csv")

random.shuffle(rows)
with open("eval_set_raw.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {len(rows)} questions to eval_set_raw.csv")