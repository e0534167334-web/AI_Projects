"""הקליינט של עוזר הגמ"ח. הרצה: streamlit run app.py"""
import html

import streamlit as st

import config
import rag
from ingest import build_index, get_collection
from models import setup_models

st.set_page_config(page_title='עוזר הגמ"ח', page_icon="📒", layout="centered")

SUGGESTIONS = [
    "כמה ערבים צריך להלוואה של 15,000 ש\"ח?",
    "מה עושים אם אי אפשר לשלם החודש?",
    "תוך כמה זמן אפשר לערער על דחייה?",
    "החזר מוקדם?",
]

# st.html ולא st.markdown: שורה ריקה בתוך <style> גורמת ל-Markdown להציג את ה-CSS כטקסט
st.html(f"<style>{(config.BASE_DIR / 'style.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource(show_spinner="מכין את מאגר המסמכים...")
def startup() -> None:
    setup_models()
    if get_collection().count() == 0:
        build_index()


def documents() -> list[tuple[str, str]]:
    """(כותרת, תוכן) לכל מסמך בתיקיית data."""
    docs = []
    for p in sorted(config.DATA_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        docs.append((text.splitlines()[0].lstrip("# ").strip(), text))
    return docs


@st.dialog("מסמך מהמאגר", width="large")
def show_document(text: str) -> None:
    st.markdown(text)


def render_sources(retrieval: rag.Retrieval) -> None:
    titles = list(dict.fromkeys(c.title for c in retrieval.chunks))
    if titles:
        st.markdown(
            f'<p class="source-line"><b>מתוך:</b> {html.escape(", ".join(titles))}</p>',
            unsafe_allow_html=True,
        )
    with st.expander("איך נמצאה התשובה"):
        if retrieval.expanded:
            st.markdown(f"**החיפוש הורחב ל:** {retrieval.search_query}")
        st.markdown(f"**הדמיון הגבוה ביותר:** {retrieval.best_similarity:.2f}")
        for c in retrieval.chunks:
            st.markdown(
                f'<div class="chunk"><b>{html.escape(c.title)} '
                f"(דמיון {c.similarity:.2f})</b>\n{html.escape(c.text)}</div>",
                unsafe_allow_html=True,
            )


def render_answer(text: str, retrieval: rag.Retrieval) -> None:
    with st.chat_message("assistant"):
        st.markdown('<span class="a-mark"></span>', unsafe_allow_html=True)
        st.markdown(text)
        render_sources(retrieval)


def show_question(question: str) -> None:
    st.markdown(f'<p class="q">{html.escape(question)}</p>', unsafe_allow_html=True)


def ask(question: str) -> None:
    show_question(question)
    try:
        with st.spinner("מחפש במסמכים..."):
            retrieval = rag.retrieve(question)
        with st.chat_message("assistant"):
            st.markdown('<span class="a-mark"></span>', unsafe_allow_html=True)
            answer = st.write_stream(rag.stream_answer(retrieval))
            render_sources(retrieval)
    except Exception as error:
        st.error("הפנייה ל-OpenAI נכשלה. בדקו את המפתח ואת היתרה בחשבון, ונסו שוב.")
        st.caption(type(error).__name__)
        return
    st.session_state.history.append((question, answer, retrieval))


st.session_state.setdefault("history", [])

st.markdown(
    """<div class="masthead"><h1>מה תרצו לדעת על הגמ"ח?</h1>
<p>התשובות מגיעות רק ממסמכי גמ"ח אור החסד, ולצד כל תשובה מופיע המקור.</p></div>""",
    unsafe_allow_html=True,
)

try:
    startup()
except RuntimeError as error:
    st.error(f"{error}. העתיקו את .env.example ל-.env והוסיפו את המפתח.")
    st.stop()

with st.sidebar:
    st.markdown("### המסמכים במאגר")
    for i, (title, text) in enumerate(documents()):
        if st.button(title, key=f"doc{i}", type="tertiary", use_container_width=True):
            show_document(text)
    st.write("")
    if st.button("בניית המאגר מחדש", use_container_width=True):
        with st.spinner("קורא את המסמכים..."):
            count = build_index()
        st.success(f"המאגר נבנה מחדש: {count} קטעים")
    if st.session_state.history and st.button("ניקוי השיחה", use_container_width=True):
        st.session_state.history = []
        st.rerun()

for question, answer, retrieval in st.session_state.history:
    show_question(question)
    render_answer(answer, retrieval)

clicked = None
if not st.session_state.history:
    left, right = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if (right if i % 2 == 0 else left).button(suggestion, use_container_width=True, key=f"s{i}"):
            clicked = suggestion

typed = st.chat_input("כתבו שאלה על הלוואות, ערבים או החזרים")
question = typed or clicked
if question:
    if clicked:
        st.session_state.pending = clicked
        st.rerun()
    ask(question)
if "pending" in st.session_state:
    ask(st.session_state.pop("pending"))
    st.rerun()
