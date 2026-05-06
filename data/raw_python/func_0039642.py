def p_statement_for(p):
    'statement : FOR LPAREN for_expr SEMI for_expr SEMI for_expr RPAREN for_statement'
    p[0] = ast.For(p[3], p[5], p[7], p[9], lineno=p.lineno(1))