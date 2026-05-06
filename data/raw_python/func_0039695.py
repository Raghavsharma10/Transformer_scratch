def p_expr_pre_incdec(p):
    '''expr : INC variable
            | DEC variable'''
    p[0] = ast.PreIncDecOp(p[1], p[2], lineno=p.lineno(1))