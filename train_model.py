"""
train_model.py
==============
Trains the Fake News Detector model.

What this script does, step by step:
1. Looks for the ISOT dataset (data/Real.csv + data/Fake.csv).
   If it instead finds a single CSV with "text" and "label" columns, it uses that.
   If NO dataset is found, it falls back to a small built-in sample so the project
   still runs — with a loud warning that accuracy will be limited.
2. Cleans the text (lowercase, remove URLs, punctuation, stopwords and agency
   tags like "(Reuters)") so the model learns WRITING PATTERNS instead of just
   memorizing which news agency wrote each article (that would be "leakage").
3. Splits 80% train / 20% test (stratified, so both classes stay balanced).
4. Trains TF-IDF + Logistic Regression and prints accuracy, precision, recall,
   F1 and a confusion matrix.
5. Saves model.pkl and vectorizer.pkl into the model/ folder using joblib.

Run it from the project root:
    python train_model.py
"""

import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (computed from this file's location so it works from any folder)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# sklearn ships a solid English stopword list, so we don't need to download NLTK.
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Regexes for "source tags" that would leak the answer. In the ISOT dataset,
# almost every REAL article starts with something like "WASHINGTON (Reuters) -".
# If we leave those in, the model "cheats" by spotting the tag, not the style.
AGENCY_TAG_RE = re.compile(
    r"\((reuters|ap|afp|ani|upi|dpa)\)\s*-?"   # "(Reuters) -" style tags
    r"|\breuters\b",                            # bare "Reuters" mentions
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Built-in sample dataset (44 diverse examples) — used ONLY when data/ is empty.
# label: 1 = REAL writing style, 0 = FAKE writing style
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    # ------------------------- FAKE-style examples (22) -------------------------
    ("SHOCKING LEAK: Senator caught accepting suitcases of cash from shadowy donors! "
     "You will not believe photo number seven. SHARE this before the corrupt media buries it!", 0),
    ("Doctors are FURIOUS: this one weird kitchen trick melts 30 pounds in three days, "
     "and Big Pharma is desperate to keep it a secret from you!", 0),
    ("BREAKING: Government insiders admit the spraying is real! Pilots finally confess to "
     "dumping chemicals over our cities. Wake up and share this before it gets deleted!", 0),
    ("EXPOSED: Famous actress is secretly not human, insiders claim! The Hollywood cover-up "
     "is crumbling as more witnesses come forward. The truth will shock you!", 0),
    ("URGENT! A leaked document proves the new law will confiscate ALL private cars by 2026. "
     "Patriots must share this before the mainstream media silences it forever!", 0),
    ("MIRACLE CURE: Grandpa, 94, reverses cancer in one week by drinking this common "
     "household liquid. Oncologists hate him! Click, share and spread the word now!", 0),
    ("BOMBSHELL: Voting machines were secretly programmed to flip votes, an anonymous "
     "whistleblower reveals stunning evidence the corrupt establishment hides from you!", 0),
    ("TERRIFYING: A mysterious illness is sweeping the nation and brave doctors who speak "
     "up are being silenced! They do not want you asking questions. Share the truth!", 0),
    ("JUST IN: World leaders met in a secret bunker to plan a total currency reset, leaked "
     "audio allegedly reveals. This is the end of freedom as we know it, folks!", 0),
    ("You will NOT believe what this teacher found hidden in a student's backpack! Police "
     "were left speechless. Detail number four will make your blood boil!", 0),
    ("HOAX EXPOSED: The moon landing footage was filmed in a desert studio, a famous "
     "director allegedly confessed on his deathbed. Share before the censors strike again!", 0),
    ("ALERT: A popular snack brand is secretly putting mind-numbing chemicals into its "
     "products, a leaked memo claims! Millions of children could be at risk. Every parent must see this!", 0),
    ("UNBELIEVABLE: A man struck by lightning gains a genius-level IQ overnight! Scientists "
     "are baffled and furious. The elite do not want this story to spread!", 0),
    ("EXCLUSIVE: A senator's secret twin brother was spotted at the capitol and the "
     "conspiracy is exploding online! The swamp runs deeper than anyone ever imagined!", 0),
    ("SHARE NOW before it is gone: lab tests allegedly show terrifying results the "
     "authorities are hiding from every family in the country. They are coming for your children next!", 0),
    ("STUNNING: An ancient pyramid discovered under the ocean proves outsiders built "
     "civilization! Archaeologists worldwide are being silenced. The history books are lies!", 0),
    ("PANIC: Banks will freeze ALL accounts next month, a supposed insider warns! Get your "
     "cash out immediately before the globalists flip the switch on all of us!", 0),
    ("DISGUSTING: A celebrity chef admits his restaurants serve lab-grown mystery meat to "
     "unsuspecting families! The horrifying details will make you sick. Share everywhere!", 0),
    ("BANNED VIDEO: A brave journalist reveals the real cause of the wildfires and it is "
     "NOT what you have been told! They are trying to erase this story forever!", 0),
    ("WOW: A local pastor claims he visited heaven and came back with a chilling warning "
     "for all Americans. The message gave witnesses chills. Pass it on before it disappears!", 0),
    ("MUST READ: A declassified file allegedly shows the real reason behind the new "
     "program, and your whole family is affected every single day. Wake up, people!", 0),
    ("INSANE: A tech billionaire is building luxury underground cities for the elite while "
     "the rest of us suffer! Leaked blueprints reveal the horrifying plan. Share the truth!", 0),

    # ------------------------- REAL-style examples (22) -------------------------
    ("City council approves 48 million dollar transit expansion after Tuesday vote. "
     "Officials said the plan adds forty electric buses and extends weekend service on six "
     "routes, with construction expected to begin next spring.", 1),
    ("Researchers at the university published a peer-reviewed study finding that moderate "
     "daily walking reduced cardiovascular risk by eighteen percent among adults over "
     "fifty, based on a ten-year cohort of twelve thousand participants.", 1),
    ("The central bank held interest rates steady on Wednesday, citing a gradual cooling in "
     "inflation, which fell to 3.1 percent last month. Economists surveyed ahead of the "
     "meeting projected one additional cut before the end of the year.", 1),
    ("Firefighters contained a brush fire near the highway on Sunday after three days of "
     "operations, authorities said. No structures were damaged, and evacuation orders for "
     "roughly two hundred homes were lifted by early evening.", 1),
    ("The ministry of health reported 1,240 new cases on Monday, a twelve percent decline "
     "from the previous week, and officials urged residents to schedule booster "
     "appointments ahead of the winter season.", 1),
    ("Parliament passed the infrastructure bill 312 to 198 on Thursday following weeks of "
     "negotiation. The legislation allocates funds for bridge repairs and broadband "
     "expansion over the next decade, the speaker's office said in a statement.", 1),
    ("A peer-reviewed trial of 3,500 patients found the new treatment reduced symptoms by "
     "27 percent compared with placebo, researchers reported at the annual cardiology "
     "conference in Chicago on Wednesday.", 1),
    ("The unemployment rate edged down to 4.1 percent in September, the labor department "
     "said Friday, as employers added 156,000 jobs. Analysts described the report as "
     "consistent with a slowly cooling labor market.", 1),
    ("Rescue crews recovered two hikers stranded on the ridge after an overnight storm, "
     "the county sheriff's office said. Both were treated for mild hypothermia at a "
     "nearby hospital and released, officials said.", 1),
    ("The school board voted five to two to extend free lunch programs through 2026, "
     "citing a district survey showing rising food insecurity among elementary students. "
     "Funding for the extension comes from a state grant.", 1),
    ("Scientists at the observatory confirmed the detection of water vapor in the "
     "atmosphere of an exoplanet roughly 120 light-years away, according to a paper "
     "published Wednesday in a peer-reviewed astronomy journal.", 1),
    ("Transit officials said the new rail line carried 41,000 riders in its first week of "
     "operation, exceeding projections by nine percent. Weekend service will begin next "
     "month, the agency said in a release.", 1),
    ("The agriculture department estimated corn yields at 177 bushels per acre this "
     "season, slightly above analyst expectations, after favorable rainfall across the "
     "Midwest in July and August.", 1),
    ("A federal judge ruled Tuesday that the agency must release its environmental review "
     "documents by the end of the month, granting a request filed earlier this year by a "
     "coalition of community groups.", 1),
    ("Hospital administrators announced the opening of a new emergency department wing on "
     "Monday, adding thirty beds and reducing average wait times, which had climbed to "
     "four hours last winter.", 1),
    ("The national team defeated its rival two to one in Saturday's qualifier, with a "
     "stoppage-time goal securing a place in next year's tournament. The coach praised "
     "the squad's composure after the match.", 1),
    ("Local police reported a nine percent decline in property crime over the past year, "
     "attributing the drop to neighborhood patrols and improved street lighting, "
     "according to annual data released Thursday.", 1),
    ("Regulators approved the merger on the condition that the combined company sell "
     "three regional assets, the commission said in a statement, ending a fourteen-month "
     "review of the deal.", 1),
    ("Archaeologists uncovered a roughly 2,000-year-old settlement during highway "
     "construction, the cultural ministry said, adding that the site would be preserved "
     "and studied over the next two years.", 1),
    ("The weather service issued a winter storm warning for northern counties, forecasting "
     "up to eight inches of snow by Wednesday morning. Schools in three districts "
     "announced delayed openings for the day.", 1),
    ("A new report from the energy agency found that solar capacity grew 22 percent last "
     "year, driven by falling panel costs and state incentives, while coal generation "
     "continued its long-term decline.", 1),
    ("Volunteers planted 15,000 native trees along the riverbank this spring as part of a "
     "multi-year restoration program, the parks department said, aiming to reduce "
     "erosion and improve water quality downstream.", 1),
]


def clean_text(raw_text):
    """Turn raw article text into simple lowercase words the model can digest.

    Order matters:
      1. lowercase everything
      2. drop URLs
      3. drop news-agency tags like "(Reuters) -"  <-- removes dataset leakage
      4. replace punctuation and digits with spaces
      5. remove stopwords and single letters
    """
    text = str(raw_text).lower()
    text = URL_RE.sub(" ", text)
    text = AGENCY_TAG_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def normalize_label(value):
    """Convert many possible label styles (1/0, REAL/FAKE, True/False) to 1 or 0."""
    # Numeric labels (including numpy/pandas numbers). NaN != NaN, so skip NaN.
    if isinstance(value, (int, float)) and value == value:
        return 1 if float(value) >= 0.5 else 0
    s = str(value).strip().lower()
    if s in ("1", "real", "true", "reliable", "legit", "real news"):
        return 1
    if s in ("0", "fake", "false", "unreliable", "clickbait", "fake news"):
        return 0
    return None


def load_dataset():
    """Find and load training data. Returns (texts, labels, source_description)."""

    # ---- Option 1: the official ISOT layout (data/Real.csv + data/Fake.csv) ----
    real_csv = os.path.join(DATA_DIR, "Real.csv")
    fake_csv = os.path.join(DATA_DIR, "Fake.csv")
    if os.path.isfile(real_csv) and os.path.isfile(fake_csv):
        real = pd.read_csv(real_csv)
        fake = pd.read_csv(fake_csv)
        texts, labels = [], []
        for df, label in ((real, 1), (fake, 0)):
            title = df["title"].fillna("") if "title" in df.columns else ""
            body = df["text"].fillna("") if "text" in df.columns else ""
            combined = (title + " " + body).astype(str)   # headline + article body
            texts.extend(combined.tolist())
            labels.extend([label] * len(df))
        print("[data] Loaded ISOT dataset: {} real + {} fake articles.".format(len(real), len(fake)))
        return texts, labels, "ISOT dataset ({} real + {} fake)".format(len(real), len(fake))

    # ---- Option 2: any single CSV in data/ that has "text" and "label" columns ----
    if os.path.isdir(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(DATA_DIR, fname)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print("[data] Skipping {}: {}".format(fname, exc))
                continue
            cols = {c.lower(): c for c in df.columns}
            if "label" not in cols or ("text" not in cols and "title" not in cols):
                continue
            texts, labels = [], []
            for _, row in df.iterrows():
                parts = [str(row[c]) for key, c in cols.items() if key in ("title", "text")]
                raw = " ".join(p for p in parts if p.lower() != "nan")
                lab = normalize_label(row[cols["label"]])
                if lab is None or not raw.strip():
                    continue
                texts.append(raw)
                labels.append(lab)
            if texts:
                print("[data] Loaded {} labeled rows from {}.".format(len(texts), fname))
                return texts, labels, "{} ({} rows)".format(fname, len(texts))

    # ---- Option 3: built-in demo data (so the project always runs) ----
    print("=" * 68)
    print(" WARNING: no dataset found in ./data")
    print(" Training on the built-in sample of {} short examples instead.".format(len(SAMPLE_DATA)))
    print(" Accuracy will be LIMITED. For real results, download the ISOT")
    print(" 'Fake and Real News Dataset' from Kaggle, put Real.csv and Fake.csv")
    print(" into the data/ folder, and run:   python train_model.py")
    print("=" * 68)
    texts = [t for t, _ in SAMPLE_DATA]
    labels = [y for _, y in SAMPLE_DATA]
    return texts, labels, "built-in sample ({} examples)".format(len(SAMPLE_DATA))


def explain_prediction(raw_text, model, vectorizer, top_n=8):
    """Return the words from THIS text that pushed the model hardest toward its verdict.

    This works because logistic regression assigns one weight to every feature:
    positive weights point toward REAL, negative weights point toward FAKE.
    We look only at words actually present in the user's text and keep the ones
    with the strongest weight in the direction of the final verdict.
    """
    features = vectorizer.transform([clean_text(raw_text)])
    decision = float(model.decision_function(features)[0])
    predicted_real = decision >= 0

    coefs = model.coef_[0]
    names = vectorizer.get_feature_names_out()
    present_cols = features.nonzero()[1]  # columns (terms) that appear in the text

    scored = [(str(names[i]), float(coefs[i])) for i in present_cols]
    # For a REAL verdict keep the biggest positive weights, otherwise the most negative.
    scored.sort(key=lambda pair: pair[1], reverse=predicted_real)

    return [
        {
            "word": word,
            "score": round(abs(score), 3),
            "pushes": "REAL" if score > 0 else "FAKE",
        }
        for word, score in scored[:top_n]
    ]


def train(save=True):
    """Full training pipeline. Returns (model, vectorizer, dataset_source)."""
    print("\n*** FAKE NEWS DETECTOR — TRAINING ***\n")

    texts, labels, source = load_dataset()
    cleaned = [clean_text(t) for t in texts]

    # 80% train / 20% test, stratified so fake/real stay balanced in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # TF-IDF: turns each article into weighted word (and two-word phrase) counts.
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),     # single words + bigrams like "share before"
        sublinear_tf=True,      # dampen very frequent words (1 + log tf)
        min_df=2 if len(X_train) > 200 else 1,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic regression: one learned weight per feature, fast, and its weights
    # double as a human-readable explanation of the decision.
    model = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    print("\n----------- TEST SET RESULTS (20% held out) -----------")
    print("Dataset            : {}".format(source))
    print("Training articles  : {}".format(len(X_train)))
    print("Test articles      : {}".format(len(X_test)))
    print("Accuracy           : {:.4f}".format(accuracy_score(y_test, predictions)))
    print("Precision          : {:.4f}".format(precision_score(y_test, predictions, zero_division=0)))
    print("Recall             : {:.4f}".format(recall_score(y_test, predictions, zero_division=0)))
    print("F1 score           : {:.4f}".format(f1_score(y_test, predictions, zero_division=0)))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("\nConfusion matrix (rows = true label, columns = predicted label):")
    print("                 pred FAKE    pred REAL")
    print("  true FAKE      {:>8}     {:>8}".format(cm[0][0], cm[0][1]))
    print("  true REAL      {:>8}     {:>8}".format(cm[1][0], cm[1][1]))
    print("\nFull classification report:")
    print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"], zero_division=0))

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(model, MODEL_PATH)
        print("[save] vectorizer -> {}".format(VECTORIZER_PATH))
        print("[save] model      -> {}".format(MODEL_PATH))

    return model, vectorizer, source


if __name__ == "__main__":
    train(save=True)
