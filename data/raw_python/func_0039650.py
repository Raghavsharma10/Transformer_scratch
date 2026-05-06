def p_statement_declare(p):
    'statement : DECLARE LPAREN declare_list RPAREN declare_statement'
    p[0] = ast.Declare(p[3], p[5], lineno=p.lineno(1))