def unicode_is_punctuation(text):
    """
    Test if a token is made entirely of Unicode characters of the following
    classes:

    - P: punctuation
    - S: symbols
    - Z: separators
    - M: combining marks
    - C: control characters

    >>> unicode_is_punctuation('word')
    False
    >>> unicode_is_punctuation('。')
    True
    >>> unicode_is_punctuation('-')
    True
    >>> unicode_is_punctuation('-3')
    False
    >>> unicode_is_punctuation('あ')
    False
    """
    for char in str_func(text):
        category = unicodedata.category(char)[0]
        if category not in 'PSZMC':
            return False
    return True