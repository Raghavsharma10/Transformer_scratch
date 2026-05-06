def p_declare_list(p):
    '''declare_list : STRING EQUALS static_scalar
                    | declare_list COMMA STRING EQUALS static_scalar'''
    if len(p) == 4:
        p[0] = [ast.Directive(p[1], p[3], lineno=p.lineno(1))]
    else:
        p[0] = p[1] + [ast.Directive(p[3], p[5], lineno=p.lineno(2))]