def p_statement_switch(p):
    'statement : SWITCH LPAREN expr RPAREN switch_case_list'
    p[0] = ast.Switch(p[3], p[5], lineno=p.lineno(1))