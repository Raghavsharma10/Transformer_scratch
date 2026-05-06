def transform_source(text):
    '''Replaces instances of

        switch expression:
    by

        for __case in _Switch(n):

    and replaces

        case expression:

    by

        if __case(expression):

    and

        default:

    by

        if __case():
    '''
    toks = tokenize.generate_tokens(StringIO(text).readline)
    result = []
    replacing_keyword = False
    for toktype, tokvalue, _, _, _ in toks:
        if toktype == tokenize.NAME and tokvalue == 'switch':
            result.extend([
                (tokenize.NAME, 'for'),
                (tokenize.NAME, '__case'),
                (tokenize.NAME, 'in'),
                (tokenize.NAME, '_Switch'),
                (tokenize.OP, '(')
            ])
            replacing_keyword = True
        elif toktype == tokenize.NAME and (tokvalue == 'case' or tokvalue == 'default'):
            result.extend([
                (tokenize.NAME, 'if'),
                (tokenize.NAME, '__case'),
                (tokenize.OP, '(')
            ])
            replacing_keyword = True
        elif replacing_keyword and tokvalue == ':':
            result.extend([
                (tokenize.OP, ')'),
                (tokenize.OP, ':')
            ])
            replacing_keyword = False
        else:
            result.append((toktype, tokvalue))
    return tokenize.untokenize(result)