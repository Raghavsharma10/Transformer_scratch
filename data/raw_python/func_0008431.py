def word_tokenize(text, tokenizer=None, include_punc=True, *args, **kwargs):
    """Convenience function for tokenizing text into words.

    NOTE: NLTK's word tokenizer expects sentences as input, so the text will be
    tokenized to sentences before being tokenized to words.

    This function returns an itertools chain object (generator).

    """
    _tokenizer = tokenizer if tokenizer is not None else NLTKPunktTokenizer()
    words = chain.from_iterable(
        WordTokenizer(tokenizer=_tokenizer).itokenize(sentence, include_punc,
                                                      *args, **kwargs)
        for sentence in sent_tokenize(text, tokenizer=_tokenizer))
    return words