"""
Phishing Email Detection Model
Project 3 — Machine Learning with Scikit-learn
Author: Your Name

Features:
  - TF-IDF vectorization (unigrams + bigrams)
  - Hand-crafted features: URL count, suspicious TLDs, urgency words,
    caps ratio, exclamation count, word count
  - Random Forest classifier (200 trees, class-balanced)
  - Accuracy, 5-fold CV, confusion matrix, classification report
"""

import re
import json
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ── Optional: plot confusion matrix if matplotlib is available ──
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Label: 1 = Phishing, 0 = Safe

emails = [
    # ── PHISHING ──────────────────────────────────────────────────────────────
    ("URGENT: Your account has been suspended! Click here immediately to verify: http://fake-bank.xyz/login?id=abc123", 1),
    ("Congratulations! You won $1,000,000! Claim your prize now: http://prize-winner.ru/claim", 1),
    ("Your PayPal account is limited. Verify now: http://paypa1-secure.tk/update", 1),
    ("ALERT: Suspicious activity on your account. Login: http://amazon-security.ga/verify", 1),
    ("Dear Customer, your Netflix subscription expired. Update payment: http://netf1ix-billing.cf/pay", 1),
    ("IRS TAX REFUND: You are owed $3,200. Claim immediately: http://irs-refund.xyz/claim", 1),
    ("Your Apple ID was used to sign in from a new device. Verify: http://apple-id-secure.tk/verify", 1),
    ("Bank Alert: Unusual login detected. Secure your account: http://chase-secure.ml/login", 1),
    ("FREE iPhone 15 giveaway! You are selected. Click: http://free-iphone.ga/win", 1),
    ("Your email storage is full. Upgrade now to avoid losing emails: http://mail-upgrade.xyz/now", 1),
    ("FINAL WARNING: Your Microsoft account will be deleted in 24 hours: http://ms-account.tk/save", 1),
    ("Password expiry notice: Reset your password now at http://secure-reset.xyz/pwd", 1),
    ("Your crypto wallet has been compromised! Secure it: http://crypto-secure.ga/fix", 1),
    ("COVID-19 relief payment ready. Collect now: http://gov-relief.xyz/covid", 1),
    ("Your DHL package is on hold. Pay customs fee: http://dhl-customs.tk/pay", 1),
    ("Verify your identity or your bank account will be frozen: http://bank-verify.ru/id", 1),
    ("Lucky Winner! Claim your $500 Amazon Gift Card: http://amazon-gift.xyz/claim", 1),
    ("Security breach detected on your account. Immediate action required: http://secure.xyz/breach", 1),
    ("Your Facebook account will be disabled unless you verify: http://fb-verify.tk/now", 1),
    ("Urgent: Confirm your Social Security number to prevent identity theft: http://ssa-verify.ga/ssn", 1),
    ("Your Google account needs verification. Click here: http://google-verify.ml/now", 1),
    ("WARNING: Your computer has virus! Download removal tool: http://antivirus-free.xyz/scan", 1),
    ("Inheritance funds of $4.5M available for you. Contact us at http://inheritance-claim.ru", 1),
    ("You have been selected for a government grant. Apply: http://gov-grant.xyz/apply", 1),
    ("Your Uber account was charged $200. Dispute now: http://uber-dispute.tk/charge", 1),

    # ── SAFE ──────────────────────────────────────────────────────────────────
    ("Hi John, just following up on our meeting from yesterday. Let me know your thoughts.", 0),
    ("Your monthly bank statement is now available. Log in at www.yourbank.com to view.", 0),
    ("Team meeting scheduled for Friday at 3 PM. Please confirm your attendance.", 0),
    ("Your order #12345 has been shipped and will arrive by Thursday.", 0),
    ("Newsletter: Top 10 Python tips for developers this month.", 0),
    ("Reminder: Your dentist appointment is tomorrow at 10 AM.", 0),
    ("Thank you for your purchase. Your receipt is attached.", 0),
    ("Project update: The development phase has been completed successfully.", 0),
    ("Your flight booking confirmation for Delta Airlines flight DL123.", 0),
    ("Welcome to our service! Here are some tips to get started.", 0),
    ("Quarterly earnings report for Q3 is now available on the investor portal.", 0),
    ("Happy Birthday! Wishing you a wonderful day from all of us.", 0),
    ("Your library book is due in 3 days. Renew online at library.org.", 0),
    ("Software update available for your device. Install at your convenience.", 0),
    ("Meeting notes from Tuesday's standup are now in the shared drive.", 0),
    ("Your subscription renewal is next month. No action needed at this time.", 0),
    ("Thanks for attending our webinar. Here is the recording link.", 0),
    ("Your tax documents for 2024 are ready for download from the IRS official website.", 0),
    ("The company picnic is scheduled for next Saturday. RSVP by Friday.", 0),
    ("Your parcel has been delivered to the front door. Tracking: 1Z999AA10123456784.", 0),
    ("Please review the attached project proposal and share your feedback.", 0),
    ("Your password was successfully changed. If this was not you, contact support.", 0),
    ("Sales report for March shows a 15% increase compared to last quarter.", 0),
    ("Reminder to submit your timesheet by end of day Friday.", 0),
    ("Your annual health insurance renewal is due. Contact HR for details.", 0),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. HAND-CRAFTED FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {".xyz", ".tk", ".ml", ".ga", ".cf", ".ru", ".gq"}
URGENCY_KEYWORDS = {
    "urgent", "immediate", "alert", "warning", "suspended", "verify",
    "claim", "won", "free", "prize", "limited", "click", "now",
    "final", "congratulations", "selected", "lucky",
}

def extract_manual_features(texts: list[str]) -> np.ndarray:
    """Return a (n_samples, 6) array of hand-crafted features."""
    rows = []
    for text in texts:
        urls = re.findall(r"http[s]?://\S+", text)
        url_count       = len(urls)
        susp_tld_count  = sum(1 for u in urls if any(t in u for t in SUSPICIOUS_TLDS))
        lower           = text.lower()
        urgency_count   = sum(1 for w in URGENCY_KEYWORDS if w in lower)
        nchars          = max(len(text), 1)
        caps_ratio      = sum(1 for c in text if c.isupper()) / nchars * 100
        exclaim_count   = text.count("!")
        word_count      = len(text.split())
        rows.append([url_count, susp_tld_count, urgency_count,
                     caps_ratio, exclaim_count, word_count])
    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────

texts  = [e[0] for e in emails]
labels = np.array([e[1] for e in emails])

X_train_txt, X_test_txt, X_train_man, X_test_man, y_train, y_test = train_test_split(
    texts,
    extract_manual_features(texts),
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

# TF-IDF on text
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train_txt)
X_test_tfidf  = tfidf.transform(X_test_txt)

# Combine TF-IDF sparse matrix with manual features
X_train = hstack([X_train_tfidf, csr_matrix(X_train_man)])
X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_man)])

print(f"Training samples : {len(y_train)}")
print(f"Test samples     : {len(y_test)}")
print(f"Total features   : {X_train.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
clf.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")

print("\n" + "=" * 60)
print("  PHISHING EMAIL DETECTION MODEL — RESULTS")
print("=" * 60)
print(f"\nTest Accuracy      : {accuracy * 100:.2f}%")
print(f"5-Fold CV Mean     : {cv_scores.mean() * 100:.2f}%  (±{cv_scores.std() * 100:.2f}%)")
print(f"\nConfusion Matrix:\n{cm}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

if HAS_MATPLOTLIB:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    disp.plot(cmap="Blues")
    plt.title("Phishing Email Detector — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("\nConfusion matrix saved to confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIVE CLASSIFIER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def classify_email(email_text: str) -> dict:
    """
    Classify a single email as Phishing or Safe.

    Returns:
        {
          "label":      "Phishing" | "Safe",
          "confidence": float (0–1),
          "signals":    list[str]
        }
    """
    manual_feat = extract_manual_features([email_text])
    tfidf_feat  = tfidf.transform([email_text])
    combined    = hstack([tfidf_feat, csr_matrix(manual_feat)])

    pred    = clf.predict(combined)[0]
    proba   = clf.predict_proba(combined)[0]
    label   = "Phishing" if pred == 1 else "Safe"
    conf    = float(proba[pred])

    # Explainability signals
    urls       = re.findall(r"http[s]?://\S+", email_text)
    susp_tlds  = [u for u in urls if any(t in u for t in SUSPICIOUS_TLDS)]
    lower      = email_text.lower()
    urgent     = [w for w in URGENCY_KEYWORDS if w in lower]
    exclaims   = email_text.count("!")

    signals = []
    if urls:       signals.append(f"Contains {len(urls)} URL(s)")
    if susp_tlds:  signals.append(f"Suspicious TLD detected: {susp_tlds[0]}")
    if urgent:     signals.append(f"Urgency keywords: {', '.join(urgent[:4])}")
    if exclaims>1: signals.append(f"{exclaims} exclamation marks")
    if not signals and label == "Safe":
        signals.append("No suspicious signals found")

    return {"label": label, "confidence": round(conf * 100, 1), "signals": signals}


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEMO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

demo_emails = [
    "URGENT: Your bank account has been suspended! Verify immediately: http://bank-secure.xyz/login",
    "Hi Sarah, just confirming our 3 PM meeting tomorrow. See you then!",
    "Congratulations! You won a free iPhone 15! Click to claim: http://free-gift.tk/claim",
    "Your order #45678 has been delivered. Thank you for shopping with us.",
    "WARNING: Unusual login to your Google account. Secure it now: http://google-verify.ml/fix",
]

print("\n" + "=" * 60)
print("  DEMO PREDICTIONS")
print("=" * 60)
for email in demo_emails:
    result = classify_email(email)
    icon   = "🚨" if result["label"] == "Phishing" else "✅"
    print(f"\n{icon} [{result['label']}]  Confidence: {result['confidence']}%")
    print(f"   Email   : {email[:80]}{'...' if len(email)>80 else ''}")
    print(f"   Signals : {' | '.join(result['signals']) or 'none'}")
