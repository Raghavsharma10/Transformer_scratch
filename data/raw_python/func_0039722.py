def p_encaps_var_dollar_curly_array_offset(p):
    'encaps_var : DOLLAR_OPEN_CURLY_BRACES STRING_VARNAME LBRACKET expr RBRACKET RBRACE'
    p[0] = ast.ArrayOffset(ast.Variable('$' + p[2], lineno=p.lineno(2)), p[4],
                           lineno=p.lineno(3))