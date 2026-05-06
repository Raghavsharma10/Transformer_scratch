def p_static_scalar(p):
    '''static_scalar : common_scalar
                     | QUOTE QUOTE
                     | QUOTE ENCAPSED_AND_WHITESPACE QUOTE'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = ''
    else:
        p[0] = p[2].decode('string_escape')