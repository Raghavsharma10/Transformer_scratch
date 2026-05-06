def compress(expression):
    """
    Compress a reference expression to group spans on the same id.
    Also works on space-separated ID lists, although a sequence of space
    characters will be considered a delimiter.

    >>> compress('a1')
    'a1'
    >>> compress('a1[3:5]')
    'a1[3:5]'
    >>> compress('a1[3:5+6:7]')
    'a1[3:5+6:7]'
    >>> compress('a1[3:5]+a1[6:7]')
    'a1[3:5+6:7]'
    >>> compress('a1 a2  a3')
    'a1 a2  a3'

    """
    tokens = []
    selection = []
    last_id = None
    for (pre, _id, _range) in robust_ref_re.findall(expression):
        if _range and _id == last_id:
            selection.extend([pre, _range])
            continue
        if selection:
            tokens.extend(selection + [']'])
            selection = []
        tokens.extend([pre, _id])
        if _range:
            selection = ['[', _range]
            last_id = _id
        else:
            last_id = None
    if selection:
        tokens.extend(selection + [']'])
    return ''.join(tokens)