def p_additional_catches(p):
    '''additional_catches : additional_catches CATCH LPAREN fully_qualified_class_name VARIABLE RPAREN LBRACE inner_statement_list RBRACE
                          | empty'''
    if len(p) == 10:
        p[0] = p[1] + [ast.Catch(p[4], ast.Variable(p[5], lineno=p.lineno(5)),
                                 p[8], lineno=p.lineno(2))]
    else:
        p[0] = []