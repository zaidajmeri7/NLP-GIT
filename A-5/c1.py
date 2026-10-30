from collections import Counter
from pathlib import Path
import math
import random
import sys


# ============================================================
# SETTINGS
# ============================================================

# Find data.txt in the same folder as this Python file
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.txt"

TOTAL_SENTENCES = 100000
DEV_SIZE = 1000
TEST_SIZE = 1000
TRAIN_SIZE = 98000

K = 0.3
SEED = 42

BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"


# ============================================================
# FIX GUJARATI OUTPUT ON WINDOWS
# ============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass


# ============================================================
# LOAD GUJARATI DATA
# ============================================================

def load_data():

    print("=" * 90)
    print("GUJARATI N-GRAM LANGUAGE MODEL")
    print("=" * 90)

    print("\nLoading dataset...")
    print("File:", DATA_FILE)

    if not DATA_FILE.exists():
        print("\nERROR: data.txt not found!")
        print("Expected location:")
        print(DATA_FILE)
        sys.exit()

    sentences = []

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                # Gujarati tokens
                tokens = line.split()

                if tokens:
                    sentences.append(tokens)

    print("Sentences found:", len(sentences))

    if len(sentences) < TOTAL_SENTENCES:

        print(
            "\nERROR: The assignment requires at least "
            "100,000 sentences."
        )

        print(
            "Your dataset contains:",
            len(sentences)
        )

        sys.exit()

    return sentences


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(sentences):

    print("\n" + "=" * 90)
    print("DATASET SPLIT")
    print("=" * 90)

    # Take exactly 100,000 sentences
    sentences = sentences[:TOTAL_SENTENCES]

    random.seed(SEED)
    random.shuffle(sentences)

    train = sentences[:TRAIN_SIZE]

    dev = sentences[
        TRAIN_SIZE:
        TRAIN_SIZE + DEV_SIZE
    ]

    test = sentences[
        TRAIN_SIZE + DEV_SIZE:
    ]

    print("Training     :", len(train))
    print("Development  :", len(dev))
    print("Test         :", len(test))

    return train, dev, test


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(train):

    print("\n" + "=" * 90)
    print("BUILDING VOCABULARY")
    print("=" * 90)

    counter = Counter()

    for sentence in train:
        counter.update(sentence)

    # Add special tokens
    counter[UNK] = 1
    counter[BOS] = 1
    counter[EOS] = 1

    vocabulary = set(counter.keys())

    print("Vocabulary size:", len(vocabulary))

    return vocabulary


# ============================================================
# N-GRAM MODEL
# ============================================================

class NGramModel:

    def __init__(self, n, train, vocabulary):

        self.n = n
        self.vocabulary = vocabulary
        self.V = len(vocabulary)

        self.ngram_counts = Counter()
        self.context_counts = Counter()

        self.total_tokens = 0

        self.train_model(train)


    # --------------------------------------------------------
    # PREPARE SENTENCE
    # --------------------------------------------------------

    def prepare_sentence(self, sentence):

        # Unknown words become <UNK>
        sentence = [
            word if word in self.vocabulary else UNK
            for word in sentence
        ]

        # Add n-1 beginning tokens
        tokens = (
            [BOS] * (self.n - 1)
            + sentence
            + [EOS]
        )

        return tokens


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train_model(self, sentences):

        for sentence in sentences:

            tokens = self.prepare_sentence(
                sentence
            )

            self.total_tokens += len(tokens)

            for i in range(
                len(tokens) - self.n + 1
            ):

                ngram = tuple(
                    tokens[
                        i:i + self.n
                    ]
                )

                self.ngram_counts[ngram] += 1

                if self.n > 1:

                    context = ngram[:-1]

                    self.context_counts[
                        context
                    ] += 1


    # --------------------------------------------------------
    # GET N-GRAMS
    # --------------------------------------------------------

    def get_ngrams(self, sentence):

        tokens = self.prepare_sentence(
            sentence
        )

        ngrams = []

        for i in range(
            len(tokens) - self.n + 1
        ):

            ngram = tuple(
                tokens[
                    i:i + self.n
                ]
            )

            ngrams.append(ngram)

        return ngrams


    # ========================================================
    # ADD-ONE / LAPLACE
    # ========================================================

    def add_one_probability(self, ngram):

        count = self.ngram_counts.get(
            ngram,
            0
        )

        if self.n == 1:

            denominator = (
                self.total_tokens
                + self.V
            )

        else:

            context = ngram[:-1]

            context_count = (
                self.context_counts.get(
                    context,
                    0
                )
            )

            denominator = (
                context_count
                + self.V
            )

        return (
            count + 1
        ) / denominator


    # ========================================================
    # ADD-K
    # ========================================================

    def add_k_probability(
        self,
        ngram,
        k=0.3
    ):

        count = self.ngram_counts.get(
            ngram,
            0
        )

        if self.n == 1:

            denominator = (
                self.total_tokens
                + k * self.V
            )

        else:

            context = ngram[:-1]

            context_count = (
                self.context_counts.get(
                    context,
                    0
                )
            )

            denominator = (
                context_count
                + k * self.V
            )

        return (
            count + k
        ) / denominator


    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        sentences,
        smoothing
    ):

        total_log_probability = 0.0
        total_words = 0

        for sentence in sentences:

            ngrams = self.get_ngrams(
                sentence
            )

            for ngram in ngrams:

                if smoothing == "add-one":

                    probability = (
                        self.add_one_probability(
                            ngram
                        )
                    )

                else:

                    probability = (
                        self.add_k_probability(
                            ngram,
                            K
                        )
                    )

                total_log_probability += math.log2(
                    probability
                )

                total_words += 1

        cross_entropy = (
            -total_log_probability
            / total_words
        )

        perplexity = (
            2 ** cross_entropy
        )

        return cross_entropy, perplexity


# ============================================================
# BUILD MODELS
# ============================================================

def build_models(train, vocabulary):

    print("\n" + "=" * 90)
    print("BUILDING LANGUAGE MODELS")
    print("=" * 90)

    models = {}

    names = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Quadgram"
    }

    for n in range(1, 5):

        print(
            f"\nBuilding {names[n]} model..."
        )

        models[n] = NGramModel(
            n,
            train,
            vocabulary
        )

        print(
            "Unique n-grams:",
            len(models[n].ngram_counts)
        )

    return models


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    models,
    dev,
    test
):

    print("\n" + "=" * 90)
    print("EVALUATING MODELS")
    print("=" * 90)

    results = []

    names = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "Quadgram"
    }

    for n in range(1, 5):

        model = models[n]

        for smoothing in [
            "add-one",
            "add-k"
        ]:

            smoothing_name = (
                "Add-One"
                if smoothing == "add-one"
                else "Add-K (K=0.3)"
            )

            print(
                f"\n{names[n]} - "
                f"{smoothing_name}"
            )

            # Development
            dev_ce, dev_pp = model.evaluate(
                dev,
                smoothing
            )

            print(
                f"Development -> "
                f"Cross Entropy: {dev_ce:.6f}, "
                f"Perplexity: {dev_pp:.6f}"
            )

            results.append(
                (
                    names[n],
                    smoothing_name,
                    "Development",
                    dev_ce,
                    dev_pp
                )
            )

            # Test
            test_ce, test_pp = model.evaluate(
                test,
                smoothing
            )

            print(
                f"Test        -> "
                f"Cross Entropy: {test_ce:.6f}, "
                f"Perplexity: {test_pp:.6f}"
            )

            results.append(
                (
                    names[n],
                    smoothing_name,
                    "Test",
                    test_ce,
                    test_pp
                )
            )

    return results


# ============================================================
# PRINT FINAL TABLE
# ============================================================

def print_final_results(results):

    print("\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(
        f"{'Model':<12}"
        f"{'Smoothing':<18}"
        f"{'Dataset':<15}"
        f"{'Cross Entropy':>20}"
        f"{'Perplexity':>20}"
    )

    print("-" * 100)

    for (
        model,
        smoothing,
        dataset,
        ce,
        pp
    ) in results:

        print(
            f"{model:<12}"
            f"{smoothing:<18}"
            f"{dataset:<15}"
            f"{ce:>20.6f}"
            f"{pp:>20.6f}"
        )

    print("=" * 100)


# ============================================================
# FIND BEST MODEL
# ============================================================

def find_best(results):

    test_results = [
        result
        for result in results
        if result[2] == "Test"
    ]

    best = min(
        test_results,
        key=lambda x: x[4]
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL ON TEST DATA")
    print("=" * 70)

    print("Model       :", best[0])
    print("Smoothing   :", best[1])
    print("Cross Entropy:", f"{best[3]:.6f}")
    print("Perplexity  :", f"{best[4]:.6f}")

    print("\nLower perplexity = better performance.")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    sentences = load_data()

    train, dev, test = split_data(
        sentences
    )

    vocabulary = build_vocabulary(
        train
    )

    models = build_models(
        train,
        vocabulary
    )

    results = evaluate_models(
        models,
        dev,
        test
    )

    print_final_results(
        results
    )

    find_best(
        results
    )

    print("\n")
    print("=" * 90)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY")
    print("=" * 90)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()