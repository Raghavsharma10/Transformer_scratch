def p_function_declaration_statement(p):
    'function_declaration_statement : FUNCTION is_reference STRING LPAREN parameter_list RPAREN LBRACE inner_statement_list RBRACE'
    p[0] = ast.Function(p[3], p[5], p[8], p[2], lineno=p.lineno(1))