def p_statement_break(p):
    '''statement : BREAK SEMI
                 | BREAK expr SEMI'''
    if len(p) == 3:
        p[0] = ast.Break(None, lineno=p.lineno(1))
    else:
        p[0] = ast.Break(p[2], lineno=p.lineno(1))