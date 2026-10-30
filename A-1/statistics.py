def corpus_statistics(tokenized_sentences):

    total_sentences = len(tokenized_sentences)

    total_words = sum(len(sentence) for sentence in tokenized_sentences)

    total_characters = sum(
        len(word)
        for sentence in tokenized_sentences
        for word in sentence
    )

    average_sentence_length = (
        total_words / total_sentences
        if total_sentences else 0
    )

    average_word_length = (
        total_characters / total_words
        if total_words else 0
    )

    unique_words = set(
        word
        for sentence in tokenized_sentences
        for word in sentence
    )

    ttr = (
        len(unique_words) / total_words
        if total_words else 0
    )

    return {
        "Total Sentences": total_sentences,
        "Total Words": total_words,
        "Total Characters": total_characters,
        "Average Sentence Length": average_sentence_length,
        "Average Word Length": average_word_length,
        "Type Token Ratio": ttr
    }