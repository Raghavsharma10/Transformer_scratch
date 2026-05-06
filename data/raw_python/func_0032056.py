def grep(target, pattern, **kwargs):
    """
    Main grep function.
    :param target: Target to apply grep on. Can be a single string, an iterable, a function, or an opened file handler.
    :param pattern: Grep pattern to search.
    :param kwargs: Optional flags (note: the docs below talk about matching 'lines', but this function also accept lists
                    and other iterables - in those cases, a 'line' means a single value from the iterable).

        The available flags are:

        - F, fixed_strings:      Interpret 'pattern' as a string or a list of strings, any of which is to be matched.
                                 If not set, will interpret 'pattern' as a python regular expression.
        - i, ignore_case:        Ignore case.
        - v, invert:             Invert (eg return non-matching lines / values).
        - w, words:              Select only those lines containing matches that form whole words.
        - x, line:               Select only matches that exactly match the whole line.
        - c, count:              Instead of the normal output, print a count of matching lines.
        - m NUM, max_count:      Stop reading after NUM matching values.
        - A NUM, after_context:  Return NUM lines of trailing context after matching lines. This will replace the string
                                 part of the reply to a list of strings. Note that in some input types this might skip
                                 following matches. For example, if the input is a file or a custom iterator.
        - B NUM, before_context: Return NUM lines of leading context before matching lines. This will replace the string
                                 part of the reply to a list of strings.
        - q, quiet:              Instead of returning string / list of strings return just a single True / False if
                                 found matches.
        - b, byte_offset:        Instead of a list of strings will return a list of (offset, string), where offset is
                                 the offset of the matched 'pattern' in line.
        - n, line_number:        Instead of a list of strings will return a list of (index, string), where index is the
                                 line number.
        - o, only_matching:      Return only the part of a matching line that matches 'pattern'.
        - r, regex_flags:        Any additional regex flags you want to add when using regex (see python re flags).
        - k, keep_eol            When iterating file, if this option is set will keep the end-of-line at the end of every
                                 line. If not (default) will trim the end of line character.
        - t, trim                Trim all whitespace characters from every line processed.

    :return: A list with matching lines (even if provided target is a single string), unless flags state otherwise.
    """
    # unify flags (convert shortcuts to full name)
    __fix_args(kwargs)

    # parse the params that are relevant to this function
    f_count = kwargs.get('count')
    f_max_count = kwargs.get('max_count')
    f_quiet = kwargs.get('quiet')

    # use the grep_iter to build the return list
    ret = []
    for value in grep_iter(target, pattern, **kwargs):

        # if quiet mode no need to continue, just return True because we got a value
        if f_quiet:
            return True

        # add current value to return list
        ret.append(value)

        # if have max limit and exceeded that limit, break:
        if f_max_count and len(ret) >= f_max_count:
            break

    # if quiet mode and got here it means we didn't find a match
    if f_quiet:
        return False

    # if requested count return results count
    if f_count:
        return len(ret)

    # return results list
    return ret