def matching_number_line_and_regex(source_lines, generated_regexes, max_line_count=15):
    """
    The first line and its number (starting with 0) in the source code that
    indicated that the source code is generated.
    :param source_lines: lines of text to scan
    :param generated_regexes: regular expressions a line must match to indicate
        the source code is generated.
    :param max_line_count: maximum number of lines to scan
    :return: a tuple of the form ``(number, line, regex)`` or ``None`` if the
        source lines do not match any ``generated_regexes``.
    """
    initial_numbers_and_lines = enumerate(itertools.islice(source_lines, max_line_count))
    matching_number_line_and_regexps = (
        (number, line, matching_regex)
        for number, line in initial_numbers_and_lines
        for matching_regex in generated_regexes
        if matching_regex.match(line)
    )
    possible_first_matching_number_line_and_regexp = list(
        itertools.islice(matching_number_line_and_regexps, 1))
    result = (possible_first_matching_number_line_and_regexp + [None])[0]
    return result