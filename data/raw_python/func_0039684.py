def p_reference_variable_string_offset(p):
    'reference_variable : reference_variable LBRACE expr RBRACE'
    p[0] = ast.StringOffset(p[1], p[3], lineno=p.lineno(2))