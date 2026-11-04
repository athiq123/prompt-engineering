"""
Customer Review Sentiment Classifier
------------------------------------
Output is ALWAYS exactly one word: Positive, Negative, or Neutral.

Two ways to use it:
  1. classify(review)      -> offline, rule-based model (no install, no API key)
  2. classify_llm(review)  -> LLM-based model (needs `pip install anthropic`
                              and the ANTHROPIC_API_KEY environment variable)

Run:  python sentiment_model.py            (runs tests, then interactive mode)
"""

import re

# ---------------------------------------------------------------------------
# 1. OFFLINE RULE-BASED MODEL
# ---------------------------------------------------------------------------

# word -> strength (1 = mild, 2 = clear, 3 = strong)
POSITIVE = {
    "good": 1, "nice": 1, "fine": 1, "decent": 1, "okay": 0, "ok": 0,
    "great": 2, "happy": 2, "helpful": 2, "love": 3, "loved": 3, "like": 1,
    "liked": 1, "satisfied": 2, "recommend": 2, "recommended": 2, "fast": 1,
    "quick": 1, "comfortable": 2, "durable": 2, "perfect": 3, "excellent": 3,
    "amazing": 3, "awesome": 3, "fantastic": 3, "wonderful": 3, "superb": 3,
    "worth": 2, "quality": 1, "beautiful": 2, "polite": 1, "friendly": 1,
    "impressed": 2, "best": 3, "smooth": 1, "reliable": 2, "easy": 1,
    "pleased": 2, "brilliant": 3, "outstanding": 3, "thanks": 1,
}

NEGATIVE = {
    "bad": 2, "poor": 2, "slow": 1, "late": 2, "delayed": 2, "damaged": 3,
    "broken": 3, "broke": 3, "cracked": 3, "defective": 3, "faulty": 3,
    "terrible": 3, "awful": 3, "horrible": 3, "worst": 3, "disappointed": 3,
    "disappointing": 3, "useless": 3, "waste": 3, "refund": 2, "return": 1,
    "returned": 2, "missing": 2, "wrong": 2, "fake": 3, "cheap": 1,
    "rude": 2, "unhelpful": 2, "problem": 2, "problems": 2, "issue": 2,
    "issues": 2, "hate": 3, "hated": 3, "regret": 3, "overpriced": 2,
    "stopped": 2, "failed": 3, "failure": 3, "scam": 3, "uncomfortable": 2,
    "dead": 3, "annoying": 2, "frustrating": 3, "unacceptable": 3,
    "never": 1, "pathetic": 3, "mediocre": 2, "lacking": 2,
}

NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't",
             "didn't", "can't", "cannot", "won't", "hardly", "barely",
             "isnt", "wasnt", "dont", "doesnt", "didnt", "cant", "wont"}
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "really": 1.5, "so": 1.3,
                "absolutely": 2.0, "totally": 1.5, "highly": 1.5}
CONTRAST = {"but", "however", "although", "though", "yet", "except"}

# Phrases that are negative even though their words look positive/neutral
NEGATIVE_PHRASES = [
    r"not worth", r"waste of money", r"doesn'?t work", r"does not work",
    r"stopped working", r"not as described", r"fell apart", r"not happy",
    r"not satisfied", r"too (small|big|slow|expensive)", r"never again",
    r"don'?t buy", r"do not buy", r"not recommend",
]
POSITIVE_PHRASES = [
    r"works? (great|perfectly|well)", r"value for money", r"worth (it|the money)",
    r"highly recommend", r"will buy again", r"exceeded (my )?expectations",
]


def _tokenize(text):
    return re.findall(r"[a-z']+|[,;.!?]", text.lower())


def _score_clause(tokens):
    """Score a list of tokens; negations flip polarity of nearby words."""
    score = 0.0
    for i, tok in enumerate(tokens):
        base = POSITIVE.get(tok, 0) - NEGATIVE.get(tok, 0)
        if base == 0:
            continue
        # intensifier directly before the word
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            base *= INTENSIFIERS[tokens[i - 1]]
        # negation within the previous 3 words flips the sign
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATIONS for w in window):
            base = -base * 0.8
        score += base
    return score


def score_review(review):
    text = review.lower().strip()
    tokens = _tokenize(text)

    # split into clauses on contrast words; the clause AFTER "but" carries
    # more weight because customers usually put their real verdict there
    clauses, current = [], []
    for tok in tokens:
        if tok in CONTRAST:
            clauses.append(current)
            current = []
        else:
            current.append(tok)
    clauses.append(current)

    total = 0.0
    for idx, clause in enumerate(clauses):
        weight = 1.0 if idx == 0 else 1.6
        total += weight * _score_clause(clause)

    # phrase-level adjustments
    for pat in NEGATIVE_PHRASES:
        if re.search(pat, text):
            total -= 2.5
    for pat in POSITIVE_PHRASES:
        if re.search(pat, text):
            total += 2.5
    return total


def classify(review: str) -> str:
    """Return exactly one of: Positive, Negative, Neutral."""
    if not isinstance(review, str) or not review.strip():
        return "Neutral"
    s = score_review(review)
    if s >= 1:
        return "Positive"
    if s <= -1:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# 2. LLM-BASED MODEL (uses the prompt from the previous step)
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are a sentiment classifier for an online shopping company.

Task: Read the customer review and classify the customer's overall sentiment.

Output rules:
- Respond with exactly one word: Positive, Negative, or Neutral.
- Do not add explanations, punctuation, quotes, or any other text.

Classification guidelines:
- Positive: satisfaction, praise, or happiness.
- Negative: dissatisfaction, complaints, problems, or regret.
- Neutral: facts or observations with no opinion or emotion.
- Mixed reviews: choose the sentiment of the stronger or more serious point.
  A major flaw (e.g. "terrible battery life") outweighs mild praise ("good").

Examples:
Review: The product is excellent, and I love its quality.
Sentiment: Positive
Review: The delivery was late, and the product was damaged.
Sentiment: Negative
Review: The product arrived yesterday.
Sentiment: Neutral
Review: The customer support team was extremely helpful.
Sentiment: Positive
Review: The product is not worth the money.
Sentiment: Negative
Review: The package contained two items.
Sentiment: Neutral
Review: The phone is good, but the battery life is terrible.
Sentiment: Negative

Now classify the following review.
Review: {review}
Sentiment:"""

VALID = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}


def classify_llm(review: str, model: str = "claude-haiku-5-5") -> str:
    """LLM classifier. Falls back to the offline model if the output is invalid."""
    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": LLM_PROMPT.format(review=review)}],
        )
        raw = msg.content[0].text.strip().strip(".\"'").lower()
        if raw in VALID:
            return VALID[raw]
    except Exception:
        pass
    return classify(review)


# ---------------------------------------------------------------------------
# 3. TESTS + INTERACTIVE MODE
# ---------------------------------------------------------------------------

TEST_CASES = [
    ("The product is excellent, and I love its quality.", "Positive"),
    ("The delivery was late, and the product was damaged.", "Negative"),
    ("The product arrived yesterday.", "Neutral"),
    ("The customer support team was extremely helpful.", "Positive"),
    ("The product is not worth the money.", "Negative"),
    ("The package contained two items.", "Neutral"),
    ("The phone is good, but the battery life is terrible.", "Negative"),
    # extra checks
    ("Fast shipping and the fabric feels great.", "Positive"),
    ("The screen cracked after two days.", "Negative"),
    ("The box has a blue label on it.", "Neutral"),
    ("Nice design, but it stopped working within a week.", "Negative"),
    ("The battery is terrible, but the camera is absolutely amazing and I love it.", "Positive"),
    ("It is not bad at all, quite good actually.", "Positive"),
    ("I ordered it on Monday.", "Neutral"),
]


def run_tests():
    passed = 0
    for review, expected in TEST_CASES:
        got = classify(review)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {got:<8} (expected {expected:<8}) | {review}")
    print(f"\n{passed}/{len(TEST_CASES)} tests passed\n")


if __name__ == "__main__":
    run_tests()
    print("Type a review (or 'quit' to exit). Output is a single word.\n")
    while True:
        try:
            text = input("Review> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            break
        print(classify(text))  # swap in classify_llm(text) to use the LLM
