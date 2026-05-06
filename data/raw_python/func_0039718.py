def p_encaps_list(p):
    '''encaps_list : encaps_list encaps_var
                   | empty'''
    if len(p) == 3:
        if p[1] == '':
            p[0] = p[2]
        else:
            p[0] = ast.BinaryOp('.', p[1], p[2], lineno=p.lineno(2))
    else:
        p[0] = ''