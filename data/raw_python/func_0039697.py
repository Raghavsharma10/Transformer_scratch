def p_expr_include(p):
    'expr : INCLUDE expr'
    p[0] = ast.Include(p[2], False, lineno=p.lineno(1))