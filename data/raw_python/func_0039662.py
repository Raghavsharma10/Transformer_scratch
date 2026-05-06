def p_global_var(p):
    '''global_var : VARIABLE
                  | DOLLAR variable
                  | DOLLAR LBRACE expr RBRACE'''
    if len(p) == 2:
        p[0] = ast.Variable(p[1], lineno=p.lineno(1))
    elif len(p) == 3:
        p[0] = ast.Variable(p[2], lineno=p.lineno(1))
    else:
        p[0] = ast.Variable(p[3], lineno=p.lineno(1))