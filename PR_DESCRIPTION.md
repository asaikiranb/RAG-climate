# Fix: Search button missing and voice search UI layout bug

## Problem summary

The RAG Climate app had a broken search experience due to **missing Search button** and **layout issues** that degraded usability for both text and voice queries.

---

## Bug description

### 1. Missing Search button (critical)

- **What:** Users had no visible Search button to explicitly trigger a search.
- **Impact:** Discoverability was poor; users relied on pressing Enter or were unsure how to run a search.
- **Expected:** A clear "Search" button next to the Mic button, consistent with common search UIs.

### 2. Layout crumpling and stacking

- **What:** Input, Mic, and Search were crammed into column layouts that caused:
  - Text to render vertically (each letter stacked) on narrow buttons
  - Elements collapsing or overlapping on smaller viewports
  - Search controls pushed below the input field instead of aligned horizontally

- **Impact:** Confusing layout and poor mobile/constrained viewport experience.

### 3. Voice + Search flow inconsistency

- **What:** Voice transcription filled the input but submission behavior was unclear; Mic and Search placement varied across attempts.
- **Impact:** Voice users were unsure whether to click Search or press Enter after transcribing.

---

## Root cause

- Use of `st.columns` with narrow ratios (e.g. `[10, 1, 1]`) for Mic and Search left insufficient space for labels.
- Multiple layout iterations separated the input and buttons across rows, causing stacking and cramped controls.
- Streamlit form constraints (Mic must be outside form to avoid accidental submit) complicated a single-row layout.

---

## Solution

### Layout change

- Single row layout: **`[Text Input | Mic | Search]`** using columns `[7, 2, 2]`.
- Both **Mic** and **Search** are `st.form_submit_button` instances so they stay on the same row and submit the form.
- Logic distinguishes which button was clicked:
  - **Mic** → toggles voice recorder (no search).
  - **Search** → runs the RAG search.

### Implementation details

- `st.form("query_form")` wraps the input, Mic, and Search.
- `col_input, col_mic, col_search = st.columns([7, 2, 2])` for a stable horizontal layout.
- Mic and Search are both form submit buttons for uniform styling and consistent UX.
- Voice transcription still populates the input; user edits if needed, then clicks Search or presses Enter.

---

## Testing

- [x] Text search: Enter or Search button both run search
- [x] Voice: Mic opens recorder → transcription fills input → Search button runs search
- [x] Layout: No vertical text or cramping on desktop and typical mobile widths
- [x] Form submit: Mic does not trigger search; Search and Enter trigger search

---

## Screenshots

Before: Cramped layout, missing or vertically stacked Search label, unclear controls.  
After: Clean row `[Input | Mic | Search]`, each element with adequate space.

---

## Related issues

- Qdrant "storage folder already accessed" error occurs when multiple Streamlit instances run; users should stop other instances before starting (documented in README / troubleshooting).
