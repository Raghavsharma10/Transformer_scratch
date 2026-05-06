def tokenize_by_number(s):
    """ splits a string into a list of tokens
        each is either a string containing no numbers
        or a float """

    r = find_number(s)

    if r == None:
        return [ s ]
    else:
        tokens = []
        if r[0] > 0:
            tokens.append(s[0:r[0]])
        tokens.append( float(s[r[0]:r[1]]) )
        if r[1] < len(s):
            tokens.extend(tokenize_by_number(s[r[1]:]))
        return tokens
    assert False