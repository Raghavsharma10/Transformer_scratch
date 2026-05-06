def basic_extractor(document, train_set):
    """A basic document feature extractor that returns a dict indicating what
    words in ``train_set`` are contained in ``document``.

    :param document: The text to extract features from. Can be a string or an iterable.
    :param list train_set: Training data set, a list of tuples of the form
        ``(words, label)``.

    """
    word_features = _get_words_from_dataset(train_set)
    tokens = _get_document_tokens(document)
    features = dict(((u'contains({0})'.format(word), (word in tokens))
                     for word in word_features))
    return features