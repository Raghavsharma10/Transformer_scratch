def p_function_call_parameter(p):
    '''function_call_parameter : expr
                               | AND variable'''
    if len(p) == 2:
        p[0] = ast.Parameter(p[1], False, lineno=p.lineno(1))
    else:
        p[0] = ast.Parameter(p[2], True, lineno=p.lineno(1))