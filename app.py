"""Misconception tutor: Streamlit front end.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.adaptive import next_question
from src.data import load_kb
from src.diagnose import Diagnoser
from src.llm import LLM
from src.mastery import StudentStore
from src.retriever import HybridIndex
from src.tutor import Tutor, chunk_lesson

st.set_page_config(page_title="Margin Notes: maths tutor", page_icon="✏️", layout="wide")

INK, RED, GREEN, PENCIL, PAPER, GRID = "#2348A8", "#C0392B", "#2F7D4F", "#6B7280", "#F6F8FA", "#E1E8F0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Kalam:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stButton button {{
  font-family: 'Atkinson Hyperlegible', 'Segoe UI', Arial, sans-serif; }}
.stApp {{
  background-color: {PAPER};
  background-image: linear-gradient({GRID} 1px, transparent 1px), linear-gradient(90deg, {GRID} 1px, transparent 1px);
  background-size: 26px 26px; }}
.block-container {{ max-width: 1100px; padding-top: 2rem; }}
h1 {{ font-weight: 700; letter-spacing: -0.01em; color: #1E2A44; margin-bottom: 0; }}
.lede {{ color: {PENCIL}; font-size: 1.05rem; max-width: 62ch; margin-top: .3rem; }}
.qcard {{ background: #fff; border-left: 4px solid {INK}; padding: 1.1rem 1.4rem; margin: .4rem 0 1rem 0; }}
.qtopic {{ color: {PENCIL}; font-size: .9rem; }}
.qtext {{ font-size: 1.6rem; font-weight: 700; color: #1E2A44; margin-top: .2rem; }}
.chosen {{ color: {INK}; font-size: 1.05rem; }}
.margin {{ font-family: 'Kalam', 'Comic Sans MS', cursive; color: {RED}; border-left: 2px solid {RED};
  padding: .2rem 0 .2rem 1rem; margin: .6rem 0; font-size: 1.15rem; line-height: 1.5; white-space: pre-wrap; }}
.margin b {{ font-weight: 700; }}
.tick {{ font-family: 'Kalam', cursive; color: {GREEN}; font-size: 1.3rem; }}
.src {{ font-size: .88rem; color: {PENCIL}; border-top: 1px dashed {GRID}; padding-top: .4rem; }}
.why {{ color: {PENCIL}; font-size: .9rem; font-style: italic; }}
</style>
""", unsafe_allow_html=True)


# ---------- shared resources ----------
@st.cache_resource
def resources():
    kb = load_kb()
    return kb, Diagnoser(kb), StudentStore()


@st.cache_resource
def tutor_for(key: str):
    kb, _, _ = resources()
    llm = LLM(api_key=key or None)
    return Tutor(kb, llm), llm


kb, diagnoser, store = resources()

with st.sidebar:
    st.subheader("Who's practising?")
    student = st.text_input("Your name", value="Demo student").strip() or "Demo student"
    language = st.selectbox("Reply language", ["English", "Hindi", "Marathi"])
    st.divider()
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:          # no secrets.toml: fine, fall back to the environment
        key = ""
    key = key or os.getenv("ANTHROPIC_API_KEY", "")
    typed = st.text_input("Anthropic API key (optional)", type="password",
                          help="Without a key, the tutor uses short template replies in English.")
    tutor, llm = tutor_for(typed or key)
    st.caption("Tutor replies: full, written by Claude" if llm.available
               else "Tutor replies: basic templates. Add an API key for full replies and Hindi or Marathi.")
    st.divider()
    st.subheader("Today's lesson")
    up = st.file_uploader("Upload class notes or a transcript (.txt)", type=["txt"])
    if up:
        st.session_state.lesson_text = up.read().decode("utf-8", errors="ignore")
        st.caption(f"Loaded {len(st.session_state.lesson_text.split())} words. Ask questions about it in the Ask tab.")

st.markdown("<h1>Margin Notes</h1>", unsafe_allow_html=True)
st.markdown("<p class='lede'>A maths tutor that works out why an answer went wrong, not just that it did. "
            "Every explanation points to the textbook.</p>", unsafe_allow_html=True)

practise, ask, teacher = st.tabs(["Practise", "Ask a question", "Class view"])


def sources_block(reply):
    if reply.sources:
        refs = "<br>".join(f"[{s.label}] {html.escape(s.title)}, {html.escape(s.ref)}" for s in reply.sources)
        st.markdown(f"<div class='src'>Where this comes from<br>{refs}</div>", unsafe_allow_html=True)


# ---------- Practise ----------
with practise:
    ss = st.session_state
    if ss.get("student") != student:
        ss.student, ss.q, ss.checked, ss.reason = student, None, False, "Starting with your weakest topic."
    if ss.q is None:
        ss.q, ss.reason = next_question(kb, store.mastery(student), store.seen(student))
        ss.checked = False
    q = ss.q
    if q is None:
        st.success(ss.reason)
        if st.button("Start again"):
            store.conn.execute("DELETE FROM attempts WHERE student=?", (student,))
            store.conn.execute("DELETE FROM mastery WHERE student=?", (student,))
            store.conn.commit()
            ss.q = None
            st.rerun()
    else:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(f"<div class='why'>{html.escape(ss.reason)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='qcard'><div class='qtopic'>{kb.concept(q['concept_id'])['name']}</div>"
                        f"<div class='qtext'>{html.escape(q['text'])}</div></div>", unsafe_allow_html=True)
            opts = list(q["options"])
            chosen = st.radio("Your answer", opts, format_func=lambda o: f"{o}.  {q['options'][o]}",
                              key=f"radio_{q['id']}", disabled=ss.checked, index=None)
            if not ss.checked:
                if st.button("Check answer", type="primary", disabled=chosen is None):
                    d = diagnoser.diagnose(q, q["options"][chosen], chosen_option=chosen)
                    ss.reply = tutor.feedback(q, chosen, d.misconception_id, language)
                    ss.diag, ss.chosen, ss.checked = d, chosen, True
                    store.record(student, q, chosen, d.misconception_id)
                    st.rerun()
            else:
                ok = ss.chosen == q["correct"]
                if ok:
                    st.markdown("<div class='tick'>✓ Correct</div>", unsafe_allow_html=True)
                else:
                    m = kb.misconception(ss.diag.misconception_id) if ss.diag.misconception_id else None
                    head = f"<b>{html.escape(m['name'])}?</b>\n" if m else ""
                    st.markdown(f"<div class='margin'>{head}{html.escape(ss.reply.text)}</div>",
                                unsafe_allow_html=True)
                sources_block(ss.reply)
                if st.button("Next question", type="primary"):
                    ss.q, ss.reason = next_question(kb, store.mastery(student), store.seen(student),
                                                    q, ss.diag.misconception_id, ok)
                    ss.checked = False
                    st.rerun()

            with st.expander("Got a different answer? Type it with your working"):
                own = st.text_input("Your answer", key=f"own_{q['id']}")
                work = st.text_area("Your working (optional)", key=f"work_{q['id']}", height=80)
                if st.button("Diagnose my answer", disabled=not own):
                    d = diagnoser.diagnose(q, own, work, llm=llm)
                    if d.misconception_id:
                        rows = [{"Possible thinking": kb.misconception(mid)["name"], "Match": f"{s:.2f}"}
                                for mid, s in d.shortlist[:3]]
                        st.markdown(f"<div class='margin'><b>{html.escape(rows[0]['Possible thinking'])}?</b>\n"
                                    f"{html.escape(kb.misconception(d.misconception_id)['probe'])}</div>",
                                    unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                        st.caption(f"Diagnosed by {d.route}; confidence {d.confidence:.0%}. "
                                   "Typed answers are matched by retrieval, so treat this as a suggestion.")

        with right:
            st.markdown("**Your topics**")
            mastery = store.mastery(student)
            if not mastery:
                st.caption("Your progress appears here after your first answer.")
            for cid, p in sorted(mastery.items(), key=lambda x: x[1]):
                name = kb.concept(cid)["name"]
                st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p:.0%} likely mastered")


# ---------- Ask ----------
with ask:
    lesson = st.session_state.get("lesson_text")
    scope = st.radio("Answer from", ["Textbook topics", "Today's lesson"], horizontal=True,
                     disabled=not lesson, help="Upload a lesson in the sidebar to ask about it.")
    lesson_index = None
    if scope == "Today's lesson" and lesson:
        lesson_index = HybridIndex(chunk_lesson(lesson), lambda d: f"{d['title']}. {d['text']}")
    hist = st.session_state.setdefault("chat", [])
    for role, content, reply in hist:
        with st.chat_message(role, avatar="✏️" if role == "user" else "📘"):
            st.write(content)
            if reply:
                sources_block(reply)
    if not hist:
        st.caption("Try: Why is 0.5 bigger than 0.125?  or  How do I expand -(x - 3)?  or type a sum such as 3/4 + 1/2")
    prompt = st.chat_input("Ask about a maths topic")
    if prompt:
        reply = tutor.answer(prompt, language, lesson_index)
        hist += [("user", prompt, None), ("assistant", reply.text, reply)]
        st.rerun()


# ---------- Class view ----------
with teacher:
    att = store.attempts()
    if att.empty:
        st.info("No answers recorded yet. Load a sample class to see how this view works, "
                "or ask students to practise first.")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Load sample class"):
        from scripts.simulate_class import simulate
        simulate()
        st.rerun()
    if c2.button("Clear class data"):
        store.reset()
        st.session_state.q = None
        st.rerun()

    if not att.empty:
        status = store.misconception_status()
        open_conf = status[(status.evidence == "confirmed") & (status.status == "open")]
        m1, m2, m3 = st.columns(3)
        m1.metric("Students", att.student.nunique())
        m2.metric("Answers recorded", len(att))
        m3.metric("Confirmed gaps still open", len(open_conf))

        st.markdown("#### Who needs help with what")
        if open_conf.empty:
            st.caption("No confirmed open misconceptions. Single wrong answers are listed as possible below.")
        else:
            tbl = open_conf.assign(Misconception=open_conf.misconception_id.map(lambda m: kb.misconception(m)["name"]),
                                   Topic=open_conf.misconception_id.map(
                                       lambda m: kb.concept(kb.misconception(m)["concept_id"])["name"]),
                                   Pattern=open_conf.times.astype(str) + " of " + open_conf.opportunities.astype(str))
            st.dataframe(tbl[["student", "Misconception", "Topic", "Pattern"]].rename(columns={"student": "Student"}),
                         hide_index=True, width="stretch")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("#### Topic mastery")
            mat = store.mastery_matrix()
            mat.columns = [kb.concept(c)["name"] for c in mat.columns]
            fig = go.Figure(go.Heatmap(
                z=mat.values, x=mat.columns, y=mat.index, zmin=0, zmax=1,
                colorscale=[[0, RED], [0.5, "#F3EDE4"], [1, GREEN]],
                hovertemplate="%{y}<br>%{x}: %{z:.0%}<extra></extra>", colorbar=dict(title="Mastery")))
            fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Atkinson Hyperlegible, sans-serif"),
                              xaxis=dict(tickangle=-40))
            st.plotly_chart(fig, width="stretch")
            st.caption("Blank cells: topic not attempted yet. Mastery is a Bayesian Knowledge Tracing estimate.")
        with right:
            st.markdown("#### Most common misconceptions")
            counts = status.groupby(["misconception_id", "evidence"]).size().unstack(fill_value=0)
            for col in ("confirmed", "possible"):
                if col not in counts:
                    counts[col] = 0
            counts = counts.assign(total=counts.sum(axis=1)).sort_values("total").tail(8)
            import textwrap
            names = ["<br>".join(textwrap.wrap(kb.misconception(m)["name"], 24)) for m in counts.index]
            bar = go.Figure([go.Bar(y=names, x=counts.confirmed, name="Confirmed pattern", orientation="h", marker_color=RED),
                             go.Bar(y=names, x=counts.possible, name="Single wrong answer", orientation="h",
                                    marker_color="#E8B4AE")])
            bar.update_layout(barmode="stack", height=430, margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", y=-0.12), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Atkinson Hyperlegible, sans-serif"),
                              xaxis_title="Students", yaxis=dict(automargin=True, tickfont=dict(size=11)))
            st.plotly_chart(bar, width="stretch")
