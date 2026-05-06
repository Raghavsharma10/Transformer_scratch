def p_expr_require_once(p):
    'expr : REQUIRE_ONCE expr'
    p[0] = ast.Require(p[2], True, lineno=p.lineno(1))