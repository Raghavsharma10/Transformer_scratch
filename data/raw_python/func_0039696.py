def p_expr_post_incdec(p):
    '''expr : variable INC
            | variable DEC'''
    p[0] = ast.PostIncDecOp(p[2], p[1], lineno=p.lineno(2))