def contains(string, substring):
    """
    Returns true if the sequence of graphemes in substring is also present in string.

    This differs from the normal python `in` operator, since the python operator will return
    true if the sequence of codepoints are withing the other string without considering
    grapheme boundaries.

    Performance notes: Very fast if `substring not in string`, since that also means that
    the same graphemes can not be in the two strings. Otherwise this function has linear time
    complexity in relation to the string length. It will traverse the sequence of graphemes until
    a match is found, so it will generally perform better for grapheme sequences that match early.

    >>> "🇸🇪" in "🇪🇸🇪🇪"
    True
    >>> grapheme.contains("🇪🇸🇪🇪", "🇸🇪")
    False
    """
    if substring not in string:
        return False

    substr_graphemes = list(graphemes(substring))

    if len(substr_graphemes) == 0:
        return True
    elif len(substr_graphemes) == 1:
        return substr_graphemes[0] in graphemes(string)
    else:
        str_iter = graphemes(string)
        str_sub_part = []
        for _ in range(len(substr_graphemes)):
            try:
                str_sub_part.append(next(str_iter))
            except StopIteration:
                return False

        for g in str_iter:
            if str_sub_part == substr_graphemes:
                return True

            str_sub_part.append(g)
            str_sub_part.pop(0)
        return str_sub_part == substr_graphemes