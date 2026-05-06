def p_class_variable_declaration_no_initial(p):
    '''class_variable_declaration : class_variable_declaration COMMA VARIABLE
                                  | VARIABLE'''
    if len(p) == 4:
        p[0] = p[1] + [ast.ClassVariable(p[3], None, lineno=p.lineno(2))]
    else:
        p[0] = [ast.ClassVariable(p[1], None, lineno=p.lineno(1))]