def p_compound_variable(p):
    '''compound_variable : VARIABLE
                         | DOLLAR LBRACE expr RBRACE'''
    if len(p) == 2:
        p[0] = ast.Variable(p[1], lineno=p.lineno(1))
    else:
        p[0] = ast.Variable(p[3], lineno=p.lineno(1))