def p_encaps_var_object_property(p):
    'encaps_var : VARIABLE OBJECT_OPERATOR STRING'
    p[0] = ast.ObjectProperty(ast.Variable(p[1], lineno=p.lineno(1)), p[3],
                              lineno=p.lineno(2))