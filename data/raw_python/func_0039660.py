def p_foreach_statement(p):
    '''foreach_statement : statement
                         | COLON inner_statement_list ENDFOREACH SEMI'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ast.Block(p[2], lineno=p.lineno(1))