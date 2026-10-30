# ============================================================
# N-GRAM LANGUAGE MODELS WITH ADD-ONE / LAPLACE SMOOTHING
# Gujarati Tokenized Dataset
# ============================================================

import sys
import random
import math
from collections import Counter


# ============================================================
# WINDOWS UTF-8 FIX
# ============================================================

# Prevent Gujarati characters from causing CP1252 encoding errors
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# 1. LOAD THE TOKENIZED DATA
# ============================================================

FILE_PATH = "tokenized_sentences.txt"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    sentences = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Total sentences:", len(sentences))

if len(sentences) < 10000:
    raise ValueError(
        "Dataset must contain at least 10,000 sentences."
    )


# ============================================================
# 2. SPLIT DATA
# ============================================================

random.seed(42)

random.shuffle(sentences)

TEST_SIZE = 1000
DEV_SIZE = 1000

test_sentences = sentences[:TEST_SIZE]

dev_sentences = sentences[
    TEST_SIZE:TEST_SIZE + DEV_SIZE
]

train_sentences = sentences[
    TEST_SIZE + DEV_SIZE:
]

print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print("Training sentences    :", len(train_sentences))
print("Development sentences :", len(dev_sentences))
print("Test sentences        :", len(test_sentences))


# ============================================================
# 3. PREPARE SENTENCES
# ============================================================

def prepare_sentence(sentence):

    tokens = sentence.split()

    if not tokens:
        return []

    return ["<BOS>"] + tokens + ["<EOS>"]


train_data = [
    prepare_sentence(sentence)
    for sentence in train_sentences
]

dev_data = [
    prepare_sentence(sentence)
    for sentence in dev_sentences
]

test_data = [
    prepare_sentence(sentence)
    for sentence in test_sentences
]


# Remove empty sentences
train_data = [
    sentence for sentence in train_data
    if sentence
]

dev_data = [
    sentence for sentence in dev_data
    if sentence
]

test_data = [
    sentence for sentence in test_data
    if sentence
]


# ============================================================
# 4. BUILD VOCABULARY
# ============================================================

vocab = set()

for sentence in train_data:
    vocab.update(sentence)

# Unknown word token
vocab.add("<UNK>")

V = len(vocab)

print("\n" + "=" * 70)
print("VOCABULARY")
print("=" * 70)

print("Vocabulary size:", V)


# ============================================================
# 5. CREATE N-GRAM COUNTERS
# ============================================================

unigram_counts = Counter()

bigram_counts = Counter()

trigram_counts = Counter()

quadrigram_counts = Counter()


# Context counts
bigram_context_counts = Counter()

trigram_context_counts = Counter()

quadrigram_context_counts = Counter()


# ============================================================
# 6. COUNT N-GRAMS
# ============================================================

for sentence in train_data:

    tokens = [
        word if word in vocab else "<UNK>"
        for word in sentence
    ]

    # --------------------------------------------------------
    # UNIGRAM
    # --------------------------------------------------------

    for word in tokens:
        unigram_counts[word] += 1


    # --------------------------------------------------------
    # BIGRAM
    # --------------------------------------------------------

    for i in range(len(tokens) - 1):

        w1 = tokens[i]
        w2 = tokens[i + 1]

        bigram_counts[(w1, w2)] += 1

        # C(w1)
        bigram_context_counts[w1] += 1


    # --------------------------------------------------------
    # TRIGRAM
    # --------------------------------------------------------

    for i in range(len(tokens) - 2):

        w1 = tokens[i]
        w2 = tokens[i + 1]
        w3 = tokens[i + 2]

        trigram_counts[
            (w1, w2, w3)
        ] += 1

        # C(w1, w2)
        trigram_context_counts[
            (w1, w2)
        ] += 1


    # --------------------------------------------------------
    # QUADRIGRAM
    # --------------------------------------------------------

    for i in range(len(tokens) - 3):

        w1 = tokens[i]
        w2 = tokens[i + 1]
        w3 = tokens[i + 2]
        w4 = tokens[i + 3]

        quadrigram_counts[
            (w1, w2, w3, w4)
        ] += 1

        # C(w1, w2, w3)
        quadrigram_context_counts[
            (w1, w2, w3)
        ] += 1


print("\n" + "=" * 70)
print("N-GRAM COUNTS")
print("=" * 70)

print("Unique unigrams    :", len(unigram_counts))
print("Unique bigrams     :", len(bigram_counts))
print("Unique trigrams    :", len(trigram_counts))
print("Unique quadrigrams :", len(quadrigram_counts))


# ============================================================
# 7. TOTAL TOKENS
# ============================================================

total_tokens = sum(
    unigram_counts.values()
)

print("\nTotal training tokens:", total_tokens)


# ============================================================
# 8. UNIGRAM PROBABILITY
# ============================================================

def unigram_probability(word):

    if word not in vocab:
        word = "<UNK>"

    numerator = (
        unigram_counts[word] + 1
    )

    denominator = (
        total_tokens + V
    )

    return numerator / denominator


# ============================================================
# 9. BIGRAM PROBABILITY
# ============================================================

def bigram_probability(w1, w2):

    if w1 not in vocab:
        w1 = "<UNK>"

    if w2 not in vocab:
        w2 = "<UNK>"

    numerator = (
        bigram_counts[(w1, w2)] + 1
    )

    denominator = (
        bigram_context_counts[w1] + V
    )

    return numerator / denominator


# ============================================================
# 10. TRIGRAM PROBABILITY
# ============================================================

def trigram_probability(w1, w2, w3):

    if w1 not in vocab:
        w1 = "<UNK>"

    if w2 not in vocab:
        w2 = "<UNK>"

    if w3 not in vocab:
        w3 = "<UNK>"

    numerator = (
        trigram_counts[
            (w1, w2, w3)
        ] + 1
    )

    denominator = (
        trigram_context_counts[
            (w1, w2)
        ] + V
    )

    return numerator / denominator


# ============================================================
# 11. QUADRIGRAM PROBABILITY
# ============================================================

def quadrigram_probability(
    w1, w2, w3, w4
):

    if w1 not in vocab:
        w1 = "<UNK>"

    if w2 not in vocab:
        w2 = "<UNK>"

    if w3 not in vocab:
        w3 = "<UNK>"

    if w4 not in vocab:
        w4 = "<UNK>"

    numerator = (
        quadrigram_counts[
            (w1, w2, w3, w4)
        ] + 1
    )

    denominator = (
        quadrigram_context_counts[
            (w1, w2, w3)
        ] + V
    )

    return numerator / denominator


# ============================================================
# 12. SAMPLE UNIGRAM PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE UNIGRAM PROBABILITIES")
print("=" * 70)

for word, count in unigram_counts.most_common(10):

    probability = unigram_probability(word)

    print(
        f"{word} | "
        f"Count = {count} | "
        f"Probability = {probability:.10f}"
    )


# ============================================================
# 13. SAMPLE BIGRAM PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE BIGRAM PROBABILITIES")
print("=" * 70)

for bigram, count in bigram_counts.most_common(10):

    probability = bigram_probability(
        bigram[0],
        bigram[1]
    )

    print(
        f"{bigram} | "
        f"Count = {count} | "
        f"Probability = {probability:.10f}"
    )


# ============================================================
# 14. SAMPLE TRIGRAM PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TRIGRAM PROBABILITIES")
print("=" * 70)

for trigram, count in trigram_counts.most_common(10):

    probability = trigram_probability(
        trigram[0],
        trigram[1],
        trigram[2]
    )

    print(
        f"{trigram} | "
        f"Count = {count} | "
        f"Probability = {probability:.10f}"
    )


# ============================================================
# 15. SAMPLE QUADRIGRAM PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE QUADRIGRAM PROBABILITIES")
print("=" * 70)

for quadrigram, count in quadrigram_counts.most_common(10):

    probability = quadrigram_probability(
        quadrigram[0],
        quadrigram[1],
        quadrigram[2],
        quadrigram[3]
    )

    print(
        f"{quadrigram} | "
        f"Count = {count} | "
        f"Probability = {probability:.10f}"
    )


# ============================================================
# 16. SENTENCE PROBABILITY
# ============================================================

def sentence_probability(
    sentence,
    model
):

    tokens = prepare_sentence(sentence)

    if not tokens:
        return 0.0

    tokens = [
        word if word in vocab else "<UNK>"
        for word in tokens
    ]

    probability = 1.0


    # --------------------------------------------------------
    # UNIGRAM
    # --------------------------------------------------------

    if model == "unigram":

        for word in tokens:

            probability *= (
                unigram_probability(word)
            )


    # --------------------------------------------------------
    # BIGRAM
    # --------------------------------------------------------

    elif model == "bigram":

        for i in range(
            len(tokens) - 1
        ):

            probability *= (
                bigram_probability(
                    tokens[i],
                    tokens[i + 1]
                )
            )


    # --------------------------------------------------------
    # TRIGRAM
    # --------------------------------------------------------

    elif model == "trigram":

        for i in range(
            len(tokens) - 2
        ):

            probability *= (
                trigram_probability(
                    tokens[i],
                    tokens[i + 1],
                    tokens[i + 2]
                )
            )


    # --------------------------------------------------------
    # QUADRIGRAM
    # --------------------------------------------------------

    elif model == "quadrigram":

        for i in range(
            len(tokens) - 3
        ):

            probability *= (
                quadrigram_probability(
                    tokens[i],
                    tokens[i + 1],
                    tokens[i + 2],
                    tokens[i + 3]
                )
            )


    else:

        raise ValueError(
            "Invalid model."
        )

    return probability


# ============================================================
# 17. TEST ONE SENTENCE
# ============================================================

example_sentence = dev_sentences[0]

print("\n" + "=" * 70)
print("EXAMPLE SENTENCE")
print("=" * 70)

print(example_sentence)

print("\nSENTENCE PROBABILITIES")

for model in [
    "unigram",
    "bigram",
    "trigram",
    "quadrigram"
]:

    probability = sentence_probability(
        example_sentence,
        model
    )

    print(
        f"{model.capitalize():12s} : "
        f"{probability:.15e}"
    )


# ============================================================
# 18. PERPLEXITY
# ============================================================

def perplexity(
    data,
    model
):

    total_log_probability = 0.0

    total_predictions = 0


    for sentence in data:

        tokens = [
            word if word in vocab else "<UNK>"
            for word in sentence
        ]


        # ----------------------------------------------------
        # UNIGRAM
        # ----------------------------------------------------

        if model == "unigram":

            for word in tokens:

                probability = (
                    unigram_probability(word)
                )

                total_log_probability += math.log(
                    probability
                )

                total_predictions += 1


        # ----------------------------------------------------
        # BIGRAM
        # ----------------------------------------------------

        elif model == "bigram":

            for i in range(
                len(tokens) - 1
            ):

                probability = (
                    bigram_probability(
                        tokens[i],
                        tokens[i + 1]
                    )
                )

                total_log_probability += math.log(
                    probability
                )

                total_predictions += 1


        # ----------------------------------------------------
        # TRIGRAM
        # ----------------------------------------------------

        elif model == "trigram":

            for i in range(
                len(tokens) - 2
            ):

                probability = (
                    trigram_probability(
                        tokens[i],
                        tokens[i + 1],
                        tokens[i + 2]
                    )
                )

                total_log_probability += math.log(
                    probability
                )

                total_predictions += 1


        # ----------------------------------------------------
        # QUADRIGRAM
        # ----------------------------------------------------

        elif model == "quadrigram":

            for i in range(
                len(tokens) - 3
            ):

                probability = (
                    quadrigram_probability(
                        tokens[i],
                        tokens[i + 1],
                        tokens[i + 2],
                        tokens[i + 3]
                    )
                )

                total_log_probability += math.log(
                    probability
                )

                total_predictions += 1


    if total_predictions == 0:
        return float("inf")


    return math.exp(
        -total_log_probability
        / total_predictions
    )


# ============================================================
# 19. DEVELOPMENT SET
# ============================================================

print("\n" + "=" * 70)
print("DEVELOPMENT SET PERPLEXITY")
print("=" * 70)

dev_results = {}

models = [
    "unigram",
    "bigram",
    "trigram",
    "quadrigram"
]

for model in models:

    pp = perplexity(
        dev_data,
        model
    )

    dev_results[model] = pp

    print(
        f"{model.capitalize():12s} : "
        f"{pp:.6f}"
    )


# ============================================================
# 20. TEST SET
# ============================================================

print("\n" + "=" * 70)
print("TEST SET PERPLEXITY")
print("=" * 70)

test_results = {}

for model in models:

    pp = perplexity(
        test_data,
        model
    )

    test_results[model] = pp

    print(
        f"{model.capitalize():12s} : "
        f"{pp:.6f}"
    )


# ============================================================
# 21. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':15s}"
    f"{'Development PP':20s}"
    f"{'Test PP':20s}"
)

print("-" * 55)

for model in models:

    print(
        f"{model.capitalize():15s}"
        f"{dev_results[model]:<20.6f}"
        f"{test_results[model]:<20.6f}"
    )


# ============================================================
# 22. BEST MODEL
# ============================================================

best_model = min(
    test_results,
    key=test_results.get
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best model:",
    best_model.capitalize()
)

print(
    "Lowest test perplexity:",
    test_results[best_model]
)


# ============================================================
# 23. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)