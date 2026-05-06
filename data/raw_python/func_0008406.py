def scan(pattern, string, *args, **kwargs):
    """ Returns True if pattern.search(Sentence(string)) may yield matches.
        If is often faster to scan prior to creating a Sentence and searching it.
    """
    return compile(pattern, *args, **kwargs).scan(string)