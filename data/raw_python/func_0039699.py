def p_expr_require(p):
    'expr : REQUIRE expr'
    p[0] = ast.Require(p[2], False, lineno=p.lineno(1))