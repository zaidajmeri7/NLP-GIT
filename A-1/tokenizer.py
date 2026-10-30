import re

def sentence_tokenizer(text):
    """
    Split text into sentences.
    Handles Gujarati danda (।), period, question mark and exclamation mark.
    """
    sentences = re.split(r'[।.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


word_pattern = re.compile(
    r'https?://\S+'
    r'|[\w.-]+@[\w.-]+\.\w+'
    r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    r'|\d+\.\d+'
    r'|\d+'
    r'|[.,!?;:"\'()\[\]{}।]'
    r'|[઀-૿]+'
)

def word_tokenizer(sentence):
    return word_pattern.findall(sentence)