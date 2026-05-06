def p_use_declaration(p):
    '''use_declaration : namespace_name
                       | NS_SEPARATOR namespace_name
                       | namespace_name AS STRING
                       | NS_SEPARATOR namespace_name AS STRING'''
    if len(p) == 2:
        p[0] = ast.UseDeclaration(p[1], None, lineno=p.lineno(1))
    elif len(p) == 3:
        p[0] = ast.UseDeclaration(p[1] + p[2], None, lineno=p.lineno(1))
    elif len(p) == 4:
        p[0] = ast.UseDeclaration(p[1], p[3], lineno=p.lineno(2))
    else:
        p[0] = ast.UseDeclaration(p[1] + p[2], p[4], lineno=p.lineno(1))