def startswith(string, prefix):
    """
    Like str.startswith, but also checks that the string starts with the given prefixes sequence of graphemes.

    str.startswith may return true for a prefix that is not visually represented as a prefix if a grapheme cluster
    is continued after the prefix ends.

    >>> grapheme.startswith("✊🏾", "✊")
    False
    >>> "✊🏾".startswith("✊")
    True
    """
    return string.startswith(prefix) and safe_split_index(string, len(prefix)) == len(prefix)