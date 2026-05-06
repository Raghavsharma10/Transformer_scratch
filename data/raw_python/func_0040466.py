def get_line_matches(input_file: str,
                     pattern: str,
                     max_occurrencies: int = 0,
                     loose_matching: bool = True) -> dict:
    r"""Get the line numbers of matched patterns.

    :parameter input_file: the file that needs to be read.
    :parameter pattern: the pattern that needs to be searched.
    :parameter max_occurrencies: the maximum number of expected occurrencies.
         Defaults to ``0`` which means that all occurrencies will be matched.
    :parameter loose_matching: ignore leading and trailing whitespace
         characters for both pattern and matched strings. Defaults to ``True``.
    :type input_file: str
    :type pattern: str
    :type max_occurrencies: int
    :type loose_matching: bool
    :returns: occurrency_matches, A dictionary where each key corresponds
         to the number of occurrencies and each value to the matched line number.
         If no match was found for that particular occurrency, the key is not
         set. This means means for example: if the first occurrency of
         pattern is at line y then: x[1] = y.
    :rtype: dict
    :raises: a built-in exception.

    .. note::
         Line numbers start from ``1``.
    """
    assert max_occurrencies >= 0

    occurrency_counter = 0.0
    occurrency_matches = dict()

    if max_occurrencies == 0:
        max_occurrencies = float('inf')
    if loose_matching:
        pattern = pattern.strip()

    line_counter = 1
    with open(input_file, 'r') as f:
        line = f.readline()
        while line and occurrency_counter < max_occurrencies:
            if loose_matching:
                line = line.strip()
            if line == pattern:
                occurrency_counter += 1.0
                occurrency_matches[int(occurrency_counter)] = line_counter
            line = f.readline()
            line_counter += 1

    return occurrency_matches