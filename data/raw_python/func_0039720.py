def p_encaps_var_array_offset(p):
    'encaps_var : VARIABLE LBRACKET encaps_var_offset RBRACKET'
    p[0] = ast.ArrayOffset(ast.Variable(p[1], lineno=p.lineno(1)), p[3],
                           lineno=p.lineno(2))