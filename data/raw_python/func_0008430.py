def sent_tokenize(text, tokenizer=None):
    """Convenience function for tokenizing sentences (not iterable).

    If tokenizer is not specified, the default tokenizer NLTKPunktTokenizer()
    is used (same behaviour as in the main `TextBlob`_ library).

    This function returns the sentences as a generator object.

    .. _TextBlob: http://textblob.readthedocs.org/

    """
    _tokenizer = tokenizer if tokenizer is not None else NLTKPunktTokenizer()
    return SentenceTokenizer(tokenizer=_tokenizer).itokenize(text)