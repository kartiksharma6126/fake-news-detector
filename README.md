# Verification Desk — Fake News Detector

🌐 **Live demo:** https://fake-news-detector-kartiksharma.onrender.com

> The demo runs on a free server, so the first visit can take up to a minute to wake up.

A full-stack college project: paste a news headline or article and get a
**REAL / FAKE** verdict with a **confidence percentage** and the **key words**
that influenced the decision.

> Built with AI assistance. I trained the model on the ISOT dataset, tested it, and deployed it as part of my college project.

**Model accuracy:** 98.85% (tested on 20% of the ISOT dataset, 8,980 articles)

> Note: High accuracy on this dataset doesn't guarantee the same result on news from other sources. This tool predicts from writing patterns and is not proof of truth.

**Stack:** Python 3.10+ · Flask · scikit-learn (TF-IDF + Logistic Regression) ·
pandas · joblib · plain HTML/CSS/JavaScript (no build step, no Node).

---

## How it works (simple version)

1. **Clean** — the text is lowercased; URLs, punctuation, stopwords and agency
   tags like “(Reuters)” are removed so the model can’t cheat by recognizing
   the source instead of the writing style.
2. **Vectorize** — TF-IDF converts the cleaned text into weighted word (and
   two-word phrase) scores: common words count little, distinctive words count a lot.
3. **Classify** — Logistic Regression has learned one weight per term from
   labeled news articles. It sums the weighted terms, converts the total into a
   probability, and returns REAL or FAKE.
4. **Explain** — because logistic regression is a linear model, we can show the
   words from your text with the strongest weights toward the final verdict.

## Project structure

```
fake-news-detector/
├── app.py                # Flask backend (routes, validation, rate limiting)
├── train_model.py        # dataset loading, cleaning, training, saving
├── test_app.py           # pytest suite for the API
├── requirements.txt      # pinned dependencies
├── Procfile              # for Render/Railway deployment (gunicorn)
├── runtime.txt           # Python version hint for Render
├── README.md
├── data/                 # <-- put Real.csv + Fake.csv here (see below)
├── model/                # model.pkl + vectorizer.pkl (created automatically)
├── templates/
│   └── index.html        # single-page frontend
└── static/
    ├── style.css
    └── script.js
```

## Dataset (optional but recommended)

The app **runs without any download** — it ships with a small built-in sample
dataset (44 examples) and will warn you in the console that accuracy is limited.

For real results, download the **ISOT “Fake and Real News Dataset”**:

- <https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets>

Then place the two files from the archive directly into the `data/` folder:

```
data/Real.csv     (columns: title, text, subject, date)
data/Fake.csv     (columns: title, text, subject, date)
```

A single CSV with `text` and `label` columns also works — just drop it into
`data/`. Labels may be `1/0`, `REAL/FAKE`, or `True/False`.

## Setup (Windows / macOS / Linux)

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows (cmd):        venv\Scripts\activate
# Windows (PowerShell): venv\Scripts\Activate.ps1
# macOS / Linux:        source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
#    NOTE: on Windows the last package (gunicorn) may fail to install.
#    That's fine — gunicorn is only needed for deployment. Delete that one
#    line from requirements.txt and install again, or just ignore the error
#    if the other packages installed.

# 3. Train the model (skip if you just want the demo sample model —
#    the app also auto-trains on first run when model/ is empty)
python train_model.py

# 4. Run the app
python app.py
#    Open http://127.0.0.1:5000
```

Development tip: `FLASK_DEBUG=1 python app.py` enables auto-reload
(on Windows cmd: `set FLASK_DEBUG=1 && python app.py`).

## Using the API directly

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Your article text here with at least twenty words so that the model has enough writing style to analyze properly.\"}"
```

Response:

```json
{
  "label": "REAL",
  "confidence": 93.4,
  "top_words": [{"word": "officials said", "score": 2.1, "pushes": "REAL"}],
  "words_analyzed": 24
}
```

Validation rules: empty text → 400, fewer than 20 words → 400,
more than 20,000 characters → 400, too many requests → 429.

## Running the tests

```bash
pytest -v
```

Covers: the home page, the health endpoint, a valid prediction, empty input,
a missing `text` key, short input, overly long input, and a non-JSON body.

## Deployment (Render / Railway)

`Procfile` (`web: gunicorn app:app`) and `runtime.txt` are included.

**Render**

1. Push this folder to a GitHub repository.
2. Render → *New → Web Service* → pick the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy.

**Railway** — same idea: *New Project → Deploy from GitHub repo*; Railway reads
the `Procfile` automatically.

**Important:** either commit the trained files in `model/` to git (simplest —
then the deployed app loads them at startup), or the app will auto-train at
boot. Auto-training on the built-in sample is instant; auto-training on the
full 45k-article ISOT dataset at boot is slow, so committing `model/` is the
better choice for production-quality results.

## Known limitations

- **Style, not truth.** The model recognizes *writing patterns* of the dataset
  (sensationalism, urgency, caps vs. attribution, numbers, measured tone).
  A true story written in clickbait style can be flagged FAKE, and a polished
  lie can be flagged REAL.
- **Domain drift.** ISOT is 2016–2018 US politics news. Slang, new topics and
  non-English text behave unpredictably.
- **Satire and opinion** are not handled as their own categories.
- **Short text** (under 20 words) is rejected on purpose — there is too little
  style to judge.
- The in-memory rate limiter resets when the server restarts and is per-process.

## Ideas for future improvement

- Train on multiple datasets and add a third class (satire / opinion).
- Add URL/domain credibility signals and metadata features (all-caps ratio,
  exclamation count) next to TF-IDF.
- Use a pretrained transformer (DistilBERT) and compare accuracy vs. speed.
- Add calibration so the confidence percentage matches real-world reliability.
- Show per-sentence highlights instead of single words.
- Redis-backed rate limiting and request logging for production use.
