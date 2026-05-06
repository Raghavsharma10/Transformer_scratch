def p_expr_unary_op(p):
    '''expr : PLUS expr
            | MINUS expr
            | NOT expr
            | BOOLEAN_NOT expr'''
    p[0] = ast.UnaryOp(p[1], p[2], lineno=p.lineno(1))