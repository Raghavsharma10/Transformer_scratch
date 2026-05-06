def p_class_constant_declaration(p):
    '''class_constant_declaration : class_constant_declaration COMMA STRING EQUALS static_scalar
                                  | CONST STRING EQUALS static_scalar'''
    if len(p) == 6:
        p[0] = p[1] + [ast.ClassConstant(p[3], p[5], lineno=p.lineno(2))]
    else:
        p[0] = [ast.ClassConstant(p[2], p[4], lineno=p.lineno(1))]