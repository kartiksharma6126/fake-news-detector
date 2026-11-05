/* ==========================================================================
   script.js — frontend logic for the Verification Desk
   Plain ES2017 (fetch + async/await). Works in Chrome, Firefox, Safari
   (including iOS), Edge and Samsung Internet.
   ========================================================================== */
(function () {
  "use strict";

  // ---------- Element references ----------
  var root = document.documentElement;
  var themeToggle = document.getElementById("theme-toggle");
  var newsInput = document.getElementById("news-input");
  var wordCount = document.getElementById("word-count");
  var charCount = document.getElementById("char-count");
  var minNote = document.getElementById("min-note");
  var analyzeBtn = document.getElementById("analyze-btn");
  var analyzeLabel = document.getElementById("analyze-label");
  var clearBtn = document.getElementById("clear-btn");
  var statusDot = document.getElementById("status-dot");
  var statusLabel = document.getElementById("status-label");
  var resultSub = document.getElementById("result-sub");
  var panelResult = document.getElementById("panel-result");
  var stateEmpty = document.getElementById("state-empty");
  var stateLoading = document.getElementById("state-loading");
  var stateError = document.getElementById("state-error");
  var stateDone = document.getElementById("state-done");
  var loadingText = document.getElementById("loading-text");
  var errorText = document.getElementById("error-text");
  var verdictStamp = document.getElementById("verdict-stamp");
  var verdictSentence = document.getElementById("verdict-sentence");
  var confidenceValue = document.getElementById("confidence-value");
  var meter = document.getElementById("meter");
  var meterFill = document.getElementById("meter-fill");
  var signalChips = document.getElementById("signal-chips");
  var reportMeta = document.getElementById("report-meta");
  var specDataset = document.getElementById("spec-dataset");
  var topbarDate = document.getElementById("topbar-date");
  var tickerTrack = document.getElementById("ticker-track");

  var MIN_WORDS = 20;
  var MAX_CHARS = 20000;

  // ---------- Example texts (each comfortably over 20 words) ----------
  var EXAMPLES = {
    "real": "The city council voted 7 to 2 on Tuesday evening to approve a $48 million " +
      "budget for public transit improvements, including new electric buses and expanded " +
      "weekend service. The measure passed after two hours of public comment, and officials " +
      "said construction could begin as early as next spring. The funding will come from a " +
      "combination of state grants and local taxes.",
    "fake-health": "SHOCKING: Doctors are FURIOUS after a leaked report claimed that a common " +
      "kitchen spice can supposedly cure diabetes practically overnight! The mainstream media " +
      "is hiding this from you because Big Pharma loses billions. SHARE this before they " +
      "delete it! Millions are already waking up to the truth that THEY do not want you to know.",
    "fake-conspiracy": "BREAKING!! Insiders confirm that the government is secretly installing " +
      "tracking chips in library cards to monitor your every move. Patriots across the country " +
      "are refusing them! Wake up, people — this is exactly what the global elites planned all " +
      "along. Spread the word before the censors silence this story forever!"
  };

  // ---------- Tiny safe localStorage helpers (private-mode friendly) ----------
  function safeGet(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function safeSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ }
  }

  // ---------- Top bar: today's date ----------
  if (topbarDate) {
    try {
      topbarDate.textContent = new Date().toLocaleDateString(undefined, {
        weekday: "long", year: "numeric", month: "long", day: "numeric"
      }).toUpperCase();
    } catch (e) { /* keep the fallback text */ }
  }

  // ---------- Theme (dark/light) with localStorage memory ----------
  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    themeToggle.setAttribute("aria-pressed", theme === "light" ? "true" : "false");
  }
  themeToggle.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    safeSet("vd-theme", next);
  });

  // ---------- Live character / word counter ----------
  function updateCounts() {
    var text = newsInput.value;
    var trimmed = text.trim();
    var words = trimmed ? trimmed.split(/\s+/).length : 0;
    wordCount.textContent = String(words);
    charCount.textContent = String(text.length);

    if (text.length > MAX_CHARS) {
      minNote.textContent = "over " + MAX_CHARS.toLocaleString() + " char limit";
      minNote.classList.remove("ok");
    } else if (words >= MIN_WORDS) {
      minNote.textContent = "length OK";
      minNote.classList.add("ok");
    } else {
      minNote.textContent = "(min " + MIN_WORDS + " words)";
      minNote.classList.remove("ok");
    }
  }
  newsInput.addEventListener("input", updateCounts);

  // ---------- "Try an example" buttons ----------
  var exampleButtons = document.querySelectorAll(".chip-btn");
  Array.prototype.forEach.call(exampleButtons, function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.getAttribute("data-example");
      if (EXAMPLES[key]) {
        newsInput.value = EXAMPLES[key];
        updateCounts();
        newsInput.focus();
      }
    });
  });

  // ---------- Result state switching ----------
  function showState(name) {
    stateEmpty.hidden = name !== "empty";
    stateLoading.hidden = name !== "loading";
    stateError.hidden = name !== "error";
    stateDone.hidden = name !== "done";
  }

  // ---------- Loading spinner: cycle status messages ----------
  var LOADING_MESSAGES = [
    "Cleaning text…",
    "Tokenizing…",
    "Comparing writing patterns…",
    "Scoring against training data…"
  ];
  var loadingTimer = null;
  function startLoadingMessages() {
    var i = 0;
    loadingText.textContent = LOADING_MESSAGES[0];
    loadingTimer = window.setInterval(function () {
      i = (i + 1) % LOADING_MESSAGES.length;
      loadingText.textContent = LOADING_MESSAGES[i];
    }, 800);
  }
  function stopLoadingMessages() {
    if (loadingTimer) { window.clearInterval(loadingTimer); loadingTimer = null; }
  }

  // ---------- Render a successful result ----------
  function renderResult(data, elapsedMs) {
    var isReal = data.label === "REAL";
    var conf = Math.round(data.confidence);

    stateDone.classList.remove("is-real", "is-fake");
    stateDone.classList.add(isReal ? "is-real" : "is-fake");
    panelResult.classList.remove("is-real", "is-fake");
    panelResult.classList.add(isReal ? "is-real" : "is-fake");

    verdictStamp.textContent = isReal ? "REAL" : "FAKE";
    // Restart the rubber-stamp animation on every new verdict.
    verdictStamp.style.animation = "none";
    void verdictStamp.offsetWidth; // force reflow to reset the animation
    verdictStamp.style.animation = "";

    verdictSentence.textContent =
      "This text reads like " + (isReal ? "REAL news" : "FAKE news") +
      " — the model is " + conf + "% confident based on its writing patterns.";

    confidenceValue.textContent = data.confidence.toFixed(1) + "%";
    meter.setAttribute("aria-label",
      "Confidence " + conf + " percent that this text is " + data.label);

    // Animate the bar from 0 to the confidence value (double rAF for Safari).
    meterFill.style.width = "0%";
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(function () {
        var clamped = Math.min(100, Math.max(0, data.confidence));
        meterFill.style.width = clamped + "%";
      });
    });

    // Top words that influenced the verdict, with a little stagger.
    signalChips.innerHTML = "";
    (data.top_words || []).forEach(function (item, i) {
      var li = document.createElement("li");
      li.style.animationDelay = (i * 70) + "ms";
      var wordSpan = document.createElement("span");
      wordSpan.textContent = item.word;
      var scoreSpan = document.createElement("span");
      scoreSpan.className = "chip-score";
      scoreSpan.textContent = item.score.toFixed(2);
      li.appendChild(wordSpan);
      li.appendChild(scoreSpan);
      signalChips.appendChild(li);
    });

    reportMeta.textContent = data.words_analyzed + " words analyzed · responded in " +
      elapsedMs + " ms · TF-IDF + Logistic Regression";
    resultSub.textContent = "Verdict: " + data.label + " · " + conf + "% confidence";
    showState("done");
  }

  // ---------- Render a friendly error ----------
  function renderError(message) {
    errorText.textContent = message;
    resultSub.textContent = "Analysis failed";
    panelResult.classList.remove("is-real", "is-fake");
    showState("error");
  }

  // ---------- Analyze: send text to POST /predict ----------
  var busy = false;
  async function analyze() {
    if (busy) return;
    var text = newsInput.value.trim();
    if (!text) {
      newsInput.focus();
      renderError("Paste some text first — or click one of the example buttons.");
      return;
    }

    busy = true;
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    analyzeLabel.textContent = "Analyzing…";
    resultSub.textContent = "Analyzing…";
    showState("loading");
    startLoadingMessages();

    var started = Date.now();
    try {
      var response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });
      var data = await response.json().catch(function () { return null; });
      var elapsed = Date.now() - started;

      if (!response.ok) {
        var msg = (data && data.error)
          ? data.error
          : "Request failed with status " + response.status + ". Please try again.";
        throw new Error(msg);
      }
      renderResult(data, elapsed);
    } catch (err) {
      var friendly = (err && err.message)
        ? err.message
        : "Could not reach the server. Is the Flask app running?";
      renderError(friendly);
    } finally {
      stopLoadingMessages();
      busy = false;
      analyzeBtn.disabled = false;
      clearBtn.disabled = false;
      analyzeLabel.textContent = "Analyze";
    }
  }
  analyzeBtn.addEventListener("click", analyze);

  // Ctrl/Cmd + Enter inside the textarea also submits
  newsInput.addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });

  // ---------- Clear button ----------
  clearBtn.addEventListener("click", function () {
    newsInput.value = "";
    updateCounts();
    panelResult.classList.remove("is-real", "is-fake");
    resultSub.textContent = "Waiting for text…";
    showState("empty");
    newsInput.focus();
  });

  // ---------- Model status badge (pings GET /health) ----------
  fetch("/health")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      statusDot.classList.add("ok");
      statusLabel.textContent = "Model ready";
      if (data && data.dataset && specDataset) {
        specDataset.textContent = data.dataset;
      }
    })
    .catch(function () {
      statusDot.classList.add("err");
      statusLabel.textContent = "Model offline";
    });

  // ---------- Ticker: duplicate content once for a seamless loop ----------
  if (tickerTrack) {
    tickerTrack.innerHTML += tickerTrack.innerHTML;
  }

  // Initial counter state
  updateCounts();
})();
