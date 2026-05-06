def p_reference_variable_array_offset(p):
    'reference_variable : reference_variable LBRACKET dim_offset RBRACKET'
    p[0] = ast.ArrayOffset(p[1], p[3], lineno=p.lineno(2))