def p_class_statement(p):
    '''class_statement : method_modifiers FUNCTION is_reference STRING LPAREN parameter_list RPAREN method_body
                       | comment
                       | variable_modifiers class_variable_declaration SEMI
                       | class_constant_declaration SEMI'''
    if len(p) == 9:
        p[0] = ast.Method(p[4], p[1], p[6], p[8], p[3], lineno=p.lineno(2))
    elif len(p) == 4:
        p[0] = ast.ClassVariables(p[1], p[2], lineno=p.lineno(3))
    elif len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ast.ClassConstants(p[1], lineno=p.lineno(2))