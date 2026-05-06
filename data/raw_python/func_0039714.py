def p_static_scalar_unary_op(p):
    '''static_scalar : PLUS static_scalar
                     | MINUS static_scalar'''
    p[0] = ast.UnaryOp(p[1], p[2], lineno=p.lineno(1))