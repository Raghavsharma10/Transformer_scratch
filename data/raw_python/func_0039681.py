def p_simple_indirect_reference(p):
    '''simple_indirect_reference : DOLLAR simple_indirect_reference
                                 | reference_variable'''
    if len(p) == 3:
        p[0] = ast.Variable(p[2], lineno=p.lineno(1))
    else:
        p[0] = p[1]