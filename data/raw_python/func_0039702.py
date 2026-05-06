def p_scalar_namespace_name(p):
    '''scalar : namespace_name
              | NS_SEPARATOR namespace_name
              | NAMESPACE NS_SEPARATOR namespace_name'''
    if len(p) == 2:
        p[0] = ast.Constant(p[1], lineno=p.lineno(1))
    elif len(p) == 3:
        p[0] = ast.Constant(p[1] + p[2], lineno=p.lineno(1))
    else:
        p[0] = ast.Constant(p[1] + p[2] + p[3], lineno=p.lineno(1))