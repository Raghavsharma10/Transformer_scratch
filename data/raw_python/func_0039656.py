def p_new_else_single(p):
    '''new_else_single : empty
                       | ELSE COLON inner_statement_list'''
    if len(p) == 4:
        p[0] = ast.Else(ast.Block(p[3], lineno=p.lineno(2)),
                        lineno=p.lineno(1))