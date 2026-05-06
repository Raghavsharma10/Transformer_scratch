def p_while_statement(p):
    '''while_statement : statement
                       | COLON inner_statement_list ENDWHILE SEMI'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ast.Block(p[2], lineno=p.lineno(1))