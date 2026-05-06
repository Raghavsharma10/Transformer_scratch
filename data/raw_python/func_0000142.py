def transform_source(text):
    '''Replaces instances of

        repeat n:
    by

        for __VAR_i in range(n):

    where __VAR_i is a string that does not appear elsewhere
    in the code sample.
    '''

    loop_keyword = 'repeat'

    nb = text.count(loop_keyword)
    if nb == 0:
        return text

    var_names = get_unique_variable_names(text, nb)

    toks = tokenize.generate_tokens(StringIO(text).readline)
    result = []
    replacing_keyword = False
    for toktype, tokvalue, _, _, _ in toks:
        if toktype == tokenize.NAME and tokvalue == loop_keyword:
            result.extend([
                (tokenize.NAME, 'for'),
                (tokenize.NAME, var_names.pop()),
                (tokenize.NAME, 'in'),
                (tokenize.NAME, 'range'),
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