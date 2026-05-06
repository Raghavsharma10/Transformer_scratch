def normalize_unicode(text):
    """
    Normalize any unicode characters to ascii equivalent
    https://docs.python.org/2/library/unicodedata.html#unicodedata.normalize
    """
    if isinstance(text, six.text_type):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
    else:
        return text