def p_expr_ternary_op(p):
    'expr : expr QUESTION expr COLON expr'
    p[0] = ast.TernaryOp(p[1], p[3], p[5], lineno=p.lineno(2))