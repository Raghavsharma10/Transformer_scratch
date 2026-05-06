def p_expr_list_assign(p):
    'expr : LIST LPAREN assignment_list RPAREN EQUALS expr'
    p[0] = ast.ListAssignment(p[3], p[6], lineno=p.lineno(1))