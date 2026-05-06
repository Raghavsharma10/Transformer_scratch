def expand(expression):
    """
    Expand a reference expression to individual spans.
    Also works on space-separated ID lists, although a sequence of space
    characters will be considered a delimiter.

    >>> expand('a1')
    'a1'
    >>> expand('a1[3:5]')
    'a1[3:5]'
    >>> expand('a1[3:5+6:7]')
    'a1[3:5]+a1[6:7]'
    >>> expand('a1 a2  a3')
    'a1 a2  a3'

    """
    tokens = []
    for (pre, _id, _range) in robust_ref_re.findall(expression):
        if not _range:
            tokens.append('{}{}'.format(pre, _id))
        else:
            tokens.append(pre)
            tokens.extend(
                '{}{}[{}:{}]'.format(delim, _id, start, end)
                for delim, start, end in span_re.findall(_range)
            )
    return ''.join(tokens)