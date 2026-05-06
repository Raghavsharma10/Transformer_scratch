def p_foreach_variable(p):
    '''foreach_variable : VARIABLE
                        | AND VARIABLE'''
    if len(p) == 2:
        p[0] = ast.ForeachVariable(p[1], False, lineno=p.lineno(1))
    else:
        p[0] = ast.ForeachVariable(p[2], True, lineno=p.lineno(1))