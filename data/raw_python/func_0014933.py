def replaceStrs(s, *args):
    r"""Replace all ``(frm, to)`` tuples in `args` in string `s`.

    >>> replaceStrs("nothing is better than warm beer",
    ...             ('nothing','warm beer'), ('warm beer','nothing'))
    'warm beer is better than nothing'

    """
    if args == (): return s
    mapping = dict((frm, to) for frm, to in args)
    return re.sub("|".join(map(re.escape, mapping.keys())),
                  lambda match:mapping[match.group(0)], s)