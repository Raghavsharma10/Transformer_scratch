def p_expr_function(p):
    'expr : FUNCTION is_reference LPAREN parameter_list RPAREN lexical_vars LBRACE inner_statement_list RBRACE'
    p[0] = ast.Closure(p[4], p[6], p[8], p[2], lineno=p.lineno(1))