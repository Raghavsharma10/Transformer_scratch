def p_top_statement_namespace(p):
    '''top_statement : NAMESPACE namespace_name SEMI
                     | NAMESPACE LBRACE top_statement_list RBRACE
                     | NAMESPACE namespace_name LBRACE top_statement_list RBRACE'''
    if len(p) == 4:
        p[0] = ast.Namespace(p[2], [], lineno=p.lineno(1))
    elif len(p) == 5:
        p[0] = ast.Namespace(None, p[3], lineno=p.lineno(1))
    else:
        p[0] = ast.Namespace(p[2], p[4], lineno=p.lineno(1))