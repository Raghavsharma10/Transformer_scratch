def endswith(string, suffix):
    """
    Like str.endswith, but also checks that the string ends with the given prefixes sequence of graphemes.

    str.endswith may return true for a suffix that is not visually represented as a suffix if a grapheme cluster
    is initiated before the suffix starts.

    >>> grapheme.endswith("🏳️‍🌈", "🌈")
    False
    >>> "🏳️‍🌈".endswith("🌈")
    True
    """
    expected_index = len(string) - len(suffix)
    return string.endswith(suffix) and safe_split_index(string, expected_index) == expected_index