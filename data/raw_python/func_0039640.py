def p_statement_while(p):
    'statement : WHILE LPAREN expr RPAREN while_statement'
    p[0] = ast.While(p[3], p[5], lineno=p.lineno(1))