def p_statement_try(p):
    'statement : TRY LBRACE inner_statement_list RBRACE CATCH LPAREN fully_qualified_class_name VARIABLE RPAREN LBRACE inner_statement_list RBRACE additional_catches'
    p[0] = ast.Try(p[3], [ast.Catch(p[7], ast.Variable(p[8], lineno=p.lineno(8)),
                                    p[11], lineno=p.lineno(5))] + p[13],
                   lineno=p.lineno(1))