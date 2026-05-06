def _split_out_of_braces(s):
    """Generator to split comma seperated string, but not split commas inside
    curly braces.

    >>> list(_split_out_of_braces("py{26, 27}-django{15, 16}, py32"))
    >>>['py{26, 27}-django{15, 16}, py32']

    """
    prev = 0
    for m in re.finditer(r"{[^{}]*}|\s*,\s*", s):
        if not m.group().startswith("{"):
            part = s[prev:m.start()]
            if part:
                yield s[prev:m.start()]
            prev = m.end()
    part = s[prev:]
    if part:
        yield part