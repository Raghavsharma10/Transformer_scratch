def create_fuzzy_pattern(pattern):
    """
    Convert a string into a fuzzy regular expression pattern.

    :param pattern: The input pattern (a string).
    :returns: A compiled regular expression object.

    This function works by adding ``.*`` between each of the characters in the
    input pattern and compiling the resulting expression into a case
    insensitive regular expression.
    """
    return re.compile(".*".join(map(re.escape, pattern)), re.IGNORECASE)