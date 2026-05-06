def _get_words_from_dataset(dataset):
    """Return a set of all words in a dataset.

    :param dataset: A list of tuples of the form ``(words, label)`` where
        ``words`` is either a string of a list of tokens.

    """
    # Words may be either a string or a list of tokens. Return an iterator
    # of tokens accordingly
    def tokenize(words):
        if isinstance(words, basestring):
            return word_tokenize(words, include_punc=False)
        else:
            return words
    all_words = chain.from_iterable(tokenize(words) for words, _ in dataset)
    return set(all_words)