def p_expr_include_once(p):
    'expr : INCLUDE_ONCE expr'
    p[0] = ast.Include(p[2], True, lineno=p.lineno(1))