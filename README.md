# Margin Notes: a misconception-diagnosing maths tutor

Most tutoring chatbots tell a student that an answer is wrong. This one works out *why*.

When a Class 8–10 student answers 3/4 + 1/2 = 4/6, the tutor recognises the pattern ("adds numerators and adds denominators"), asks a guiding question instead of revealing the answer, explains the rule from the textbook with a citation, and then serves a question designed to check whether that specific gap has closed. A teacher view shows which misconceptions are spreading through the class and which students hold them consistently.

Built by Mrugaja Churi.

## How it works

```
 student answer ──► Diagnose ──────────────► Tutor reply ──────────► Next question
                    │                         │                       │
                    ├ tagged distractor?      ├ retrieves the concept  ├ targets the same
                    │   → direct lookup       │   note (NCERT topic)   │   misconception if one
                    └ typed answer?           ├ Socratic probe         │   was found
                        → hybrid retrieval    └ cites every source     └ else weakest topic
                          over misconception                            (BKT mastery)
                          library + topic prior
                                    │
                                    ▼
                         Student model (SQLite): attempts, mastery, misconception status
                                    │
                                    ▼
                         Class view: mastery heatmap, confirmed gaps, common errors
```

**Retrieval.** A hybrid index fuses BM25 over words with TF-IDF over character n-grams using reciprocal rank fusion. Character n-grams matter here because maths notation such as `1/4`, `x^2` and `-(` is broken apart by ordinary word tokenisers.

**Structural query expansion.** A question such as `3/4 + 1/2 = ?` contains almost no words, so lexical retrieval cannot tell which topic it belongs to. Transparent rules in `src/features.py` read the shape of the expression (fraction plus fraction, a power, a bracket with a minus sign) and append topic vocabulary to the query. This raised topic-unknown diagnosis from 32% to 63% top-1 accuracy.

**Diagnosis.** In multiple-choice mode, every wrong option in the question bank is tagged with the misconception it reveals, so the diagnosis is a direct lookup. For typed answers, the question, answer and working are matched against a misconception library, weighted by a topic prior. An optional LLM re-ranker reorders the shortlist.

**Student model.** Mastery per topic uses Bayesian Knowledge Tracing (Corbett and Anderson, 1994). A misconception is marked *confirmed* only when the error recurs on at least half of the questions that tested for it, which separates a consistent pattern from a slip. It is marked *cleared* when the student later answers a question targeting it correctly.

**Grounding and refusal.** Replies cite their sources. Ask mode declines questions that share no subject vocabulary with the course material, instead of answering from general knowledge.

## Results on the seed data

Run `python -m eval.evaluate`. Each tagged wrong option is diagnosed with its tag hidden, from the question text and chosen answer alone.

| Setting | Cases | Top-1 | Top-3 | MAP@25 |
|---|---|---|---|---|
| Topic known (inside a practice session) | 41 | 73.2% | 100.0% | 0.862 |
| Topic unknown (any wrong answer) | 41 | 63.4% | 90.2% | 0.766 |

Ask mode: 10/10 in-scope questions answered with the correct top source; 6/6 off-syllabus questions declined.

Class simulation (`python -m scripts.simulate_class`): 12 simulated students with planted misconceptions. 21 of 22 planted misconceptions were surfaced among those the students were actually tested on.

**Read these numbers with care.** The seed bank is small (33 questions, 24 misconceptions) and self-authored, and the expansion rules were developed while looking at it, so these figures show the pipeline works rather than how it generalises. Two known limitations: within one topic, the retriever sometimes confuses sibling misconceptions (for example the two fraction-addition errors), and with only two or three questions per misconception, "confirmed" flags in the simulation are often triggered by students who simply don't know the topic. Both improve with more data, which is what the Eedi ingestion below is for.

## Quick start

```bash
git clone https://github.com/mrugaja-0702/Mathematics-Tutor.git
cd Mathematics-Tutor
pip install -r requirements.txt
streamlit run app.py
```

The app runs fully offline with template replies. For full Socratic replies written by Claude, replies in Hindi or Marathi, and LLM re-ranking of typed answers, set an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or paste it into the sidebar
export TUTOR_MODEL=claude-sonnet-5-5     # optional
```

In the app, open **Class view** and press **Load sample class** to populate the dashboard.

## Scaling to real data

**NCERT textbooks.** Download chapter PDFs from ncert.nic.in into `data/raw/ncert/` and run `python -m src.ingest_ncert`. Chunks keep their file and page number, so citations point to an exact textbook page.

**Eedi misconception dataset.** The Kaggle competition *Eedi – Mining Misconceptions in Mathematics* provides thousands of real multiple-choice questions where wrong options are labelled by experts. Place `train.csv` and `misconception_mapping.csv` in `data/raw/eedi/` and run `python -m src.ingest_eedi`. The script holds out 20% of questions and builds misconception signatures only from the rest, so evaluation on the held-out set is free of leakage. `eval/evaluate.py` detects the split automatically.

## Project structure

```
app.py                  Streamlit app: Practise, Ask a question, Class view
src/retriever.py        Hybrid BM25 + character n-gram index with RRF
src/features.py         Structural query expansion for maths expressions
src/diagnose.py         Tagged lookup and retrieval diagnosis
src/tutor.py            Grounded feedback and Ask mode with refusal
src/mastery.py          SQLite store, Bayesian Knowledge Tracing, misconception status
src/adaptive.py         Next-question selection
src/llm.py              Optional Anthropic wrapper with offline fallback
src/ingest_ncert.py     NCERT PDF ingestion
src/ingest_eedi.py      Eedi dataset ingestion with held-out split
data/seed/              Concepts, misconception library, tagged question bank
eval/evaluate.py        Diagnosis and Ask-mode evaluation
scripts/simulate_class.py  Simulated class for the teacher view
tests/                  Unit tests (pytest)
```

## Deploying

Push to GitHub, then create an app on Streamlit Community Cloud pointing at `app.py`. Add `ANTHROPIC_API_KEY` under the app's secrets if you want full replies. The SQLite database resets when the hosted app restarts, which is acceptable for a demo; a hosted database would be needed for real classroom use.

## Notes on content

Concept notes in `data/seed/` are original explanations written for this project. The NCERT references point to the topic in the textbook series and should be checked against the current edition, since the Class 6–8 books have been revised since 2024. After NCERT ingestion, citations come from the actual pages.

The tutor is a study aid. It is designed to send students back to the textbook and their teacher, not to replace either.
