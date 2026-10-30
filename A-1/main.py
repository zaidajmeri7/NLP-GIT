from tokenizer import sentence_tokenizer, word_tokenizer
from statistics import corpus_statistics

# Read Gujarati corpus
with open("A-1/Data/gu.txt", "r", encoding="utf-8") as file:
    text = file.read(10 * 1024 * 1024)  # Read first 10 MB

# Sentence tokenization
sentences = sentence_tokenizer(text)

# Word tokenization
tokenized_sentences = []

for sentence in sentences:
    tokenized_sentences.append(word_tokenizer(sentence))

# Save tokenized sentences
with open("tokenized_sentences.txt", "w", encoding="utf-8") as file:

    for sentence in tokenized_sentences:
        file.write(" ".join(sentence))
        file.write("\n")

# Corpus statistics
stats = corpus_statistics(tokenized_sentences)

print("\nCorpus Statistics\n")

for key, value in stats.items():
    print(f"{key}: {value}")