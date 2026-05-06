def is_url(text):
    """ Check if the given text looks like a URL. """
    if text is None:
        return False
    text = text.lower()
    return text.startswith('http://') or text.startswith('https://') or \
        text.startswith('urn:') or text.startswith('file://')