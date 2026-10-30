"""
NLP Lab Assignment - Gujarati Language Model
----------------------------------------------
1. Build Unigram, Bigram, Trigram and Quadgram language models.
2. Split at least 100,000 sentences into:
      Development set : 1,000 sentences
      Test set        : 1,000 sentences
      Training set    : remaining sentences
3. Evaluate every model with:
      - Add-One / Laplace smoothing
      - Add-K smoothing, K = 0.3
4. Report cross-entropy and perplexity on development and test sets.

INPUT:
    Put the tokenized language data from Assignment-1 in "data.txt".
    The file should contain one sentence per line.

If your Assignment-1 data is already tokenized, the code keeps the tokens.
If punctuation is attached to words, simple whitespace tokenization is used.
"""

from collections import Counter
import math
import random
import re
from pathlib import Path
import sys

# Force UTF-8 output for Gujarati text on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data.txt"

# IMPORTANT FOR GUJARATI DATA:
# Save data.txt as UTF-8. Python 3 reads Gujarati Unicode correctly.
# No transliteration or English conversion is required.

TOTAL_REQUIRED_SENTENCES = 100_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000

K = 0.3
SEED = 42

# Special tokens
BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"


# ============================================================
# DATA LOADING
# ============================================================

def tokenize_sentence(line):
    """
    Tokenization for Assignment-1 style data.
    If your data is already tokenized by spaces, this is sufficient.
    """
    line = line.strip()

    if not line:
        return []

    # Gujarati is fully supported by Python 3 / UTF-8.
    # We keep whitespace-separated tokens exactly as they appear.
    # This works with Gujarati text such as:
    # "ગુજરાતી ભાષા ખૂબ સુંદર છે"
    tokens = line.split()

    return tokens


def load_sentences(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"\nERROR: {filename} was not found.\n"
            f"Create {filename} and put your Assignment-1 tokenized "
            f"language data in it, one sentence per line."
        )

    sentences = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = tokenize_sentence(line)

            if tokens:
                sentences.append(tokens)

    return sentences


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(sentences):
    """
    Use exactly the first 100,000 non-empty sentences if more are
    available. Randomly split them into train/dev/test.

    Dev = 1,000
    Test = 1,000
    Train = remaining 98,000
    """

    if len(sentences) < TOTAL_REQUIRED_SENTENCES:
        raise ValueError(
            f"\nOnly {len(sentences):,} sentences were found.\n"
            f"The assignment requires at least "
            f"{TOTAL_REQUIRED_SENTENCES:,} sentences."
        )

    # Use 100,000 sentences as required.
    sentences = sentences[:TOTAL_REQUIRED_SENTENCES]

    random.seed(SEED)
    random.shuffle(sentences)

    dev = sentences[:DEV_SIZE]
    test = sentences[DEV_SIZE:DEV_SIZE + TEST_SIZE]
    train = sentences[DEV_SIZE + TEST_SIZE:]

    return train, dev, test


# ============================================================
# VOCABULARY
# ============================================================

def build_vocabulary(train_sentences):
    """
    Build vocabulary from training data only.
    Gujarati Unicode text is fully supported.
    """

    vocab = Counter()

    for sentence in train_sentences:
        vocab.update(sentence)

    # Counter does not have .add().
    # Add special tokens by assigning a count.
    vocab[UNK] = max(vocab.get(UNK, 0), 1)
    vocab[BOS] = max(vocab.get(BOS, 0), 1)
    vocab[EOS] = max(vocab.get(EOS, 0), 1)

    return set(vocab.keys())


def replace_unknowns(sentence, vocabulary):
    return [
        word if word in vocabulary else UNK
        for word in sentence
    ]


# ============================================================
# N-GRAM MODEL
# ============================================================

class NGramModel:
    def __init__(self, n, train_sentences, vocabulary):
        self.n = n
        self.vocabulary = vocabulary
        self.V = len(vocabulary)

        # n-gram counts
        self.ngram_counts = Counter()

        # (n-1)-gram/context counts
        self.context_counts = Counter()

        self.train_token_count = 0

        self.train(train_sentences)

    def add_boundary_tokens(self, sentence):
        """
        For:
            Unigram  -> words + EOS
            Bigram   -> <s> words EOS
            Trigram  -> <s> <s> words EOS
            Quadgram -> <s> <s> <s> words EOS

        We add n-1 BOS tokens.
        """

        sentence = replace_unknowns(sentence, self.vocabulary)

        return [BOS] * (self.n - 1) + sentence + [EOS]

    def train(self, sentences):
        for sentence in sentences:
            tokens = self.add_boundary_tokens(sentence)

            self.train_token_count += len(tokens)

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.ngram_counts[ngram] += 1

                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

    # --------------------------------------------------------
    # ADD-ONE / LAPLACE
    # --------------------------------------------------------

    def probability_add_one(self, ngram):
        """
        P(w_n | w_1 ... w_(n-1))
        =
        (C(w_1,...,w_n) + 1)
        /
        (C(w_1,...,w_(n-1)) + V)

        For unigram:
        (C(w) + 1) / (N + V)
        """

        count = self.ngram_counts.get(tuple(ngram), 0)

        if self.n == 1:
            denominator = self.train_token_count + self.V
        else:
            context = tuple(ngram[:-1])
            denominator = self.context_counts.get(context, 0) + self.V

        return (count + 1.0) / denominator

    # --------------------------------------------------------
    # ADD-K
    # --------------------------------------------------------

    def probability_add_k(self, ngram, k=K):
        """
        P(w_n | context)
        =
        (C(ngram) + k)
        /
        (C(context) + kV)

        For unigram:
        (C(w) + k) / (N + kV)
        """

        count = self.ngram_counts.get(tuple(ngram), 0)

        if self.n == 1:
            denominator = self.train_token_count + k * self.V
        else:
            context = tuple(ngram[:-1])
            denominator = self.context_counts.get(context, 0) + k * self.V

        return (count + k) / denominator

    # --------------------------------------------------------
    # SENTENCE PROBABILITY / PERPLEXITY
    # --------------------------------------------------------

    def get_ngrams_for_sentence(self, sentence):
        tokens = self.add_boundary_tokens(sentence)

        ngrams = []

        for i in range(len(tokens) - self.n + 1):
            ngrams.append(tuple(tokens[i:i + self.n]))

        return ngrams

    def evaluate(self, sentences, smoothing="add-one"):
        """
        Returns:
            total log probability
            number of predicted tokens
            cross entropy
            perplexity
        """

        total_log_probability = 0.0
        total_predictions = 0

        for sentence in sentences:
            ngrams = self.get_ngrams_for_sentence(sentence)

            for ngram in ngrams:

                if smoothing == "add-one":
                    probability = self.probability_add_one(ngram)
                elif smoothing == "add-k":
                    probability = self.probability_add_k(ngram, K)
                else:
                    raise ValueError("Unknown smoothing method.")

                total_log_probability += math.log2(probability)
                total_predictions += 1

        if total_predictions == 0:
            return 0.0, 0.0, float("inf")

        cross_entropy = -total_log_probability / total_predictions
        perplexity = 2 ** cross_entropy

        return (
            total_log_probability,
            cross_entropy,
            perplexity
        )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_result_table(results):
    print("\n" + "=" * 92)
    print("LANGUAGE MODEL RESULTS")
    print("=" * 92)

    print(
        f"{'Model':<12}"
        f"{'Smoothing':<15}"
        f"{'Set':<10}"
        f"{'Cross-Entropy':>18}"
        f"{'Perplexity':>18}"
    )

    print("-" * 92)

    for row in results:
        print(
            f"{row['model']:<12}"
            f"{row['smoothing']:<15}"
            f"{row['set']:<10}"
            f"{row['cross_entropy']:>18.6f}"
            f"{row['perplexity']:>18.6f}"
        )

    print("=" * 92)


def print_sample_probabilities(models):
    """
    Print a few example probabilities to demonstrate that
    smoothing gives non-zero probabilities even to unseen n-grams.
    """

    print("\n" + "=" * 92)
    print("SAMPLE SMOOTHED PROBABILITIES")
    print("=" * 92)

    # Gujarati examples. These are only demonstration n-grams;
    # they do not affect model training or evaluation.
    examples = {
        1: [("ગુજરાતી",)],
        2: [("ગુજરાતી", "ભાષા")],
        3: [("ગુજરાતી", "ભાષા", "ખૂબ")],
        4: [("ગુજરાતી", "ભાષા", "ખૂબ", "સુંદર")]
    }

    for n, model in models.items():
        print(f"\n{n}-GRAM MODEL")

        for ngram in examples[n]:

            p1 = model.probability_add_one(ngram)
            pk = model.probability_add_k(ngram, K)

            print(f"  N-gram: {ngram}")
            print(f"    Add-One : {p1:.10f}")
            print(f"    Add-K   : {pk:.10f}")

    print("=" * 92)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "#" * 92)
    print("NLP LAB ASSIGNMENT - N-GRAM LANGUAGE MODELS")
    print("#" * 92)

    print("\n[1] Loading tokenized data...")

    sentences = load_sentences(DATA_FILE)

    print(f"    Sentences found : {len(sentences):,}")

    print("\n[2] Preparing train/dev/test sets...")

    train, dev, test = prepare_data(sentences)

    print(f"    Training set    : {len(train):,} sentences")
    print(f"    Development set : {len(dev):,} sentences")
    print(f"    Test set        : {len(test):,} sentences")

    print("\n[3] Building vocabulary from training data...")

    vocabulary = build_vocabulary(train)

    print(f"    Vocabulary size : {len(vocabulary):,}")

    print("\n[4] Building four language models...")

    models = {}

    for n in range(1, 5):

        names = {
            1: "Unigram",
            2: "Bigram",
            3: "Trigram",
            4: "Quadgram"
        }

        print(f"    Building {names[n]} model...")

        models[n] = NGramModel(
            n=n,
            train_sentences=train,
            vocabulary=vocabulary
        )

        print(
            f"      Unique {n}-grams: "
            f"{len(models[n].ngram_counts):,}"
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\n[5] Applying Add-One / Laplace smoothing...")
    print("    Applying Add-K smoothing with K = 0.3...")

    results = []

    names = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Quadgram"
    }

    smoothing_methods = [
        ("Add-One", "add-one"),
        ("Add-K (K=0.3)", "add-k")
    ]

    datasets = [
        ("Development", dev),
        ("Test", test)
    ]

    for n, model in models.items():

        for smoothing_name, smoothing_code in smoothing_methods:

            for dataset_name, dataset in datasets:

                _, cross_entropy, perplexity = model.evaluate(
                    dataset,
                    smoothing=smoothing_code
                )

                results.append({
                    "model": names[n],
                    "smoothing": smoothing_name,
                    "set": dataset_name,
                    "cross_entropy": cross_entropy,
                    "perplexity": perplexity
                })

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print_result_table(results)

    print_sample_probabilities(models)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 92)
    print("BEST MODELS ON TEST SET")
    print("=" * 92)

    test_results = [
        r for r in results
        if r["set"] == "Test"
    ]

    test_results.sort(key=lambda x: x["perplexity"])

    for i, row in enumerate(test_results, 1):
        print(
            f"{i}. {row['model']:<10} | "
            f"{row['smoothing']:<13} | "
            f"Perplexity = {row['perplexity']:.6f}"
        )

    print("\nLower perplexity = better language-model performance.")

    print("\n" + "#" * 92)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY")
    print("#" * 92)


if __name__ == "__main__":
    main()
