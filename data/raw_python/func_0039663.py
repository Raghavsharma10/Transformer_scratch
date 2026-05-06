def p_static_var(p):
    '''static_var : VARIABLE EQUALS static_scalar
                  | VARIABLE'''
    if len(p) == 4:
        p[0] = ast.StaticVariable(p[1], p[3], lineno=p.lineno(1))
    else:
        p[0] = ast.StaticVariable(p[1], None, lineno=p.lineno(1))