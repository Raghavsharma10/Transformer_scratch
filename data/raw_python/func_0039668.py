def p_class_variable_declaration_initial(p):
    '''class_variable_declaration : class_variable_declaration COMMA VARIABLE EQUALS static_scalar
                                  | VARIABLE EQUALS static_scalar'''
    if len(p) == 6:
        p[0] = p[1] + [ast.ClassVariable(p[3], p[5], lineno=p.lineno(2))]
    else:
        p[0] = [ast.ClassVariable(p[1], p[3], lineno=p.lineno(1))]