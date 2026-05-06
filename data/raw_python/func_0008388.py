def contains_extractor(document):
    """A basic document feature extractor that returns a dict of words that the
    document contains."""
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features