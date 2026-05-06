def p_statement_continue(p):
    '''statement : CONTINUE SEMI
                 | CONTINUE expr SEMI'''
    if len(p) == 3:
        p[0] = ast.Continue(None, lineno=p.lineno(1))
    else:
        p[0] = ast.Continue(p[2], lineno=p.lineno(1))