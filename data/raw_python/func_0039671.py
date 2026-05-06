def p_parameter(p):
    '''parameter : VARIABLE
                 | class_name VARIABLE
                 | ARRAY VARIABLE
                 | AND VARIABLE
                 | class_name AND VARIABLE
                 | ARRAY AND VARIABLE
                 | VARIABLE EQUALS static_scalar
                 | class_name VARIABLE EQUALS static_scalar
                 | ARRAY VARIABLE EQUALS static_scalar
                 | AND VARIABLE EQUALS static_scalar
                 | class_name AND VARIABLE EQUALS static_scalar
                 | ARRAY AND VARIABLE EQUALS static_scalar'''
    if len(p) == 2: # VARIABLE
        p[0] = ast.FormalParameter(p[1], None, False, None, lineno=p.lineno(1))
    elif len(p) == 3 and p[1] == '&': # AND VARIABLE
        p[0] = ast.FormalParameter(p[2], None, True, None, lineno=p.lineno(1))
    elif len(p) == 3 and p[1] != '&': # STRING VARIABLE
        p[0] = ast.FormalParameter(p[2], None, False, p[1], lineno=p.lineno(1))
    elif len(p) == 4 and p[2] != '&': # VARIABLE EQUALS static_scalar
        p[0] = ast.FormalParameter(p[1], p[3], False, None, lineno=p.lineno(1))
    elif len(p) == 4 and p[2] == '&': # STRING AND VARIABLE
        p[0] = ast.FormalParameter(p[3], None, True, p[1], lineno=p.lineno(1))
    elif len(p) == 5 and p[1] == '&': # AND VARIABLE EQUALS static_scalar
        p[0] = ast.FormalParameter(p[2], p[4], True, None, lineno=p.lineno(1))
    elif len(p) == 5 and p[1] != '&': # STRING VARIABLE EQUALS static_scalar
        p[0] = ast.FormalParameter(p[2], p[4], False, p[1], lineno=p.lineno(1))
    else: # STRING AND VARIABLE EQUALS static_scalar
        p[0] = ast.FormalParameter(p[3], p[5], True, p[1], lineno=p.lineno(1))