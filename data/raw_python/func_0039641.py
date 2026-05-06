def p_statement_do_while(p):
    'statement : DO statement WHILE LPAREN expr RPAREN SEMI'
    p[0] = ast.DoWhile(p[2], p[5], lineno=p.lineno(1))