"""כל ההחלטות הארכיטקטוניות של הפרויקט במקום אחד."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "storage"
COLLECTION_NAME = "gemach_docs"

# חלוקה לפי טוקנים עם חפיפה
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# מודלים (אפשר להחליף דרך קובץ .env)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# חיפוש: MMR שולף FETCH_K מועמדים ובוחר מתוכם TOP_K
TOP_K = 5
FETCH_K = 15
MMR_THRESHOLD = 0.7  # 1 = רלוונטיות בלבד, 0 = גיוון בלבד

# מתחת לסף הזה הקטע נחשב לא קשור לשאלה
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

# הרחבת שאלה רק לשאלות קצרות מהמספר הזה של מילים
SHORT_QUESTION_WORDS = 5

NO_ANSWER = "המידע הזה לא מופיע במסמכי הגמ\"ח. אפשר לפנות למשרד בטלפון 03-5550123."
