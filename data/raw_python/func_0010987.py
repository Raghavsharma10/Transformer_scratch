def zero_width_split(pattern, string):
    """
    Split a string on a regex that only matches zero-width strings

    :param pattern: Regex pattern that matches zero-width strings
    :param string: String to split on.
    :return: Split array
    """
    splits = list((m.start(), m.end()) for m in regex.finditer(pattern, string, regex.VERBOSE))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits] + [len(string)]
    return [string[start:end] for start, end in zip(starts, ends)]