def p_declare_statement(p):
    '''declare_statement : statement
                         | COLON inner_statement_list ENDDECLARE SEMI'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ast.Block(p[2], lineno=p.lineno(1))