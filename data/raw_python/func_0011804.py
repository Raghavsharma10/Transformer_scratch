def selections(expression, keep_delimiters=True):
    """
    Split the expression into individual selection expressions. The
    delimiters will be kept as separate items if keep_delimters=True.
    Also works on space-separated ID lists, although a sequence of space
    characters will be considered a delimiter.

    >>> selections('a1')
    ['a1']
    >>> selections('a1[3:5]')
    ['a1[3:5]']
    >>> selections('a1[3:5+6:7]')
    ['a1[3:5+6:7]']
    >>> selections('a1[3:5+6:7]+a2[1:4]')
    ['a1[3:5+6:7]', '+', 'a2[1:4]']
    >>> selections('a1[3:5+6:7]+a2[1:4]', keep_delimiters=False)
    ['a1[3:5+6:7]', 'a2[1:4]']
    >>> selections('a1 a2  a3')
    ['a1', ' ', 'a2', '  ', 'a3']

    """
    tokens = []
    for (pre, _id, _range) in robust_ref_re.findall(expression):
        if keep_delimiters and pre:
            tokens.append(pre)
        if _id:
            if _range:
                tokens.append('{}[{}]'.format(_id, _range))
            else:
                tokens.append(_id)
    return tokens