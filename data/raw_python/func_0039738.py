def terms_from_dict(source):
    """ Convert a dict representing a query to a string.

        Args:
            source -- A dict with query xpaths as keys and text or nested query dicts as values.

        Returns:
            A string composed from the nested query terms given.

    >>> terms_from_dict({'document': {'title': "Title this is", 'text': "A long text."}})
    '<document><text>A long text.</text><title>Title this is</title></document>'

    >>> terms_from_dict({'document/title': "Title this is", 'document/text': "A long text."})
    '<document><title>Title this is</title></document><document><text>A long text.</text></document>'
    """
    parsed = ''
    for xpath, text in source.items():
        if hasattr(text, 'keys'):
            parsed += term(terms_from_dict(text), xpath, escape=False)
        else:
            parsed += term(text, xpath)
    return parsed