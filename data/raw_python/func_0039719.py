def p_encaps_list_string(p):
    'encaps_list : encaps_list ENCAPSED_AND_WHITESPACE'
    if p[1] == '':
        p[0] = p[2].decode('string_escape')
    else:
        p[0] = ast.BinaryOp('.', p[1], p[2].decode('string_escape'),
                            lineno=p.lineno(2))