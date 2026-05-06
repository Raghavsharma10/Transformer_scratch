def p_expr_exit(p):
    '''expr : EXIT
            | EXIT LPAREN RPAREN
            | EXIT LPAREN expr RPAREN'''
    if len(p) == 5:
        p[0] = ast.Exit(p[3], lineno=p.lineno(1))
    else:
        p[0] = ast.Exit(None, lineno=p.lineno(1))