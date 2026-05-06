def p_statement_return(p):
    '''statement : RETURN SEMI
                 | RETURN expr SEMI'''
    if len(p) == 3:
        p[0] = ast.Return(None, lineno=p.lineno(1))
    else:
        p[0] = ast.Return(p[2], lineno=p.lineno(1))