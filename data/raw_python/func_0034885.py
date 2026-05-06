def expand_curlys(s):
    """Takes string and returns list of options:

    Example
    -------
    >>> expand_curlys("py{26, 27}")
    ["py26", "py27"]

    """
    from functools import reduce
    curleys = list(re.finditer(r"{[^{}]*}", s))
    return reduce(_replace_curly, reversed(curleys), [s])