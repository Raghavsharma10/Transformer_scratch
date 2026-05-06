def p_expr_assign(p):
    '''expr : variable EQUALS expr
            | variable EQUALS AND expr'''
    if len(p) == 5:
        p[0] = ast.Assignment(p[1], p[4], True, lineno=p.lineno(2))
    else:
        p[0] = ast.Assignment(p[1], p[3], False, lineno=p.lineno(2))