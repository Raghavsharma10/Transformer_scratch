def guess_encoding(text, default=DEFAULT_ENCODING):
    """Guess string encoding.

    Given a piece of text, apply character encoding detection to
    guess the appropriate encoding of the text.
    """
    result = chardet.detect(text)
    return normalize_result(result, default=default)