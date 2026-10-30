import json
import math



# 1. LOAD DATASET


with open("Assignment_2/text_segmentation_dataset.json", "r") as f:
    data = json.load(f)

word_counts = data["word_counts"]
test_cases = data["test_cases"]

vocabulary = set(word_counts.keys())
total_words = sum(word_counts.values())

# Longest word in vocabulary
MAX_WORD_LENGTH = max(len(word) for word in vocabulary)


# 2. GREEDY LONGEST-WORD APPROACH

def greedy_segmentation(text):
    """
    At every position, choose the longest word
    present in the vocabulary.
    """

    result = []
    i = 0

    while i < len(text):

        longest_word = None

        # Try longest possible word first
        max_length = min(MAX_WORD_LENGTH, len(text) - i)

        for length in range(max_length, 0, -1):

            word = text[i:i + length]

            if word in vocabulary:
                longest_word = word
                break

        # If no word is found, keep the character
        # as an unknown token.
        if longest_word is None:
            longest_word = text[i]

        result.append(longest_word)

        i += len(longest_word)

    return result


# 3. DYNAMIC PROGRAMMING APPROACH


def dp_segmentation(text):
    """
    Finds the segmentation with maximum log probability.

    P(word) = word_frequency / total_corpus_words

    log P(word) = log(word_frequency) - log(total_words)

    Dynamic programming maximizes the sum of log probabilities.
    """

    n = len(text)

    # dp[i] = maximum log probability for text[:i]
    dp = [-math.inf] * (n + 1)

    # previous[i] stores:
    # (previous_position, word)
    previous = [None] * (n + 1)

    dp[0] = 0.0

    for i in range(n):

        if dp[i] == -math.inf:
            continue

        # Try every possible word beginning at position i
        max_length = min(MAX_WORD_LENGTH, n - i)

        for length in range(1, max_length + 1):

            word = text[i:i + length]

            if word in word_counts:

                # Log probability
                log_probability = (
                    math.log(word_counts[word])
                    - math.log(total_words)
                )

                new_score = dp[i] + log_probability

                if new_score > dp[i + length]:
                    dp[i + length] = new_score
                    previous[i + length] = (i, word)

    # If no complete segmentation was possible
    if dp[n] == -math.inf:
        return greedy_segmentation(text)

    
    # Backtrack to recover the words
    

    words = []
    position = n

    while position > 0:

        prev_position, word = previous[position]

        words.append(word)

        position = prev_position

    words.reverse()

    return words


# 4. EDIT DISTANCE


def edit_distance(predicted, actual):
    """
    Levenshtein edit distance between two lists of words.

    Operations:
        - insertion
        - deletion
        - substitution
    """

    m = len(predicted)
    n = len(actual)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if predicted[i - 1] == actual[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[m][n]



# 5. EVALUATION


greedy_total_correct = 0
dp_total_correct = 0

total_actual_words = 0

greedy_edit_distance = 0
dp_edit_distance = 0

greedy_exact_matches = 0
dp_exact_matches = 0


for test_case in test_cases:

    text = test_case["input"]
    actual = test_case["ground_truth"].split()

    # Predictions
    greedy_prediction = greedy_segmentation(text)
    dp_prediction = dp_segmentation(text)

    
    # Accuracy
    

    # Compare words at corresponding positions
    greedy_correct = sum(
        p == a
        for p, a in zip(greedy_prediction, actual)
    )

    dp_correct = sum(
        p == a
        for p, a in zip(dp_prediction, actual)
    )

    greedy_total_correct += greedy_correct
    dp_total_correct += dp_correct

    total_actual_words += len(actual)

    
    # Edit distance
    

    greedy_edit_distance += edit_distance(
        greedy_prediction,
        actual
    )

    dp_edit_distance += edit_distance(
        dp_prediction,
        actual
    )

    # Exact complete segmentation
    if greedy_prediction == actual:
        greedy_exact_matches += 1

    if dp_prediction == actual:
        dp_exact_matches += 1



# 6. FINAL METRICSf


greedy_accuracy = (
    greedy_total_correct / total_actual_words
) * 100

dp_accuracy = (
    dp_total_correct / total_actual_words
) * 100

greedy_avg_edit_distance = (
    greedy_edit_distance / len(test_cases)
)

dp_avg_edit_distance = (
    dp_edit_distance / len(test_cases)
)

greedy_exact_accuracy = (
    greedy_exact_matches / len(test_cases)
) * 100

dp_exact_accuracy = (
    dp_exact_matches / len(test_cases)
) * 100



# 7. DISPLAY RESULTS


print("=" * 60)
print("TEXT SEGMENTATION RESULTS")
print("=" * 60)

print("\nNumber of test cases:", len(test_cases))
print("Vocabulary size:", len(vocabulary))

print("\n--- GREEDY APPROACH ---")
print("Word Accuracy:",
      round(greedy_accuracy, 2), "%")
print("Average Edit Distance:",
      round(greedy_avg_edit_distance, 3))
print("Exact Segmentation Accuracy:",
      round(greedy_exact_accuracy, 2), "%")

print("\n--- DYNAMIC PROGRAMMING APPROACH ---")
print("Word Accuracy:",round(dp_accuracy, 2), "%")
print("Average Edit Distance:",round(dp_avg_edit_distance, 3))
print("Exact Segmentation Accuracy:",round(dp_exact_accuracy, 2), "%")



# 8. SHOW SOME EXAMPLES


print("\n" + "=" * 60)
print("SAMPLE RESULTS")
print("=" * 60)

for test_case in test_cases[:10]:

    text = test_case["input"]
    actual = test_case["ground_truth"]

    greedy_result = " ".join(
        greedy_segmentation(text)
    )

    dp_result = " ".join(
        dp_segmentation(text)
    )

    print("\nInput:")
    print(text)

    print("\nGround Truth:")
    print(actual)

    print("\nGreedy:")
    print(greedy_result)

    print("\nDynamic Programming:")
    print(dp_result)

    print("-" * 60)