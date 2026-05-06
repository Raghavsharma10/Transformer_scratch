def slugify(text, sep='-'):
    """A simple slug generator."""
    text = stringify(text)
    if text is None:
        return None
    text = text.replace(sep, WS)
    text = normalize(text, ascii=True)
    if text is None:
        return None
    return text.replace(WS, sep)