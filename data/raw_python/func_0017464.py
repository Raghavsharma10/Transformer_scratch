def p_expression_group(p):
    '''expression : LPAREN expression RPAREN
                  | LSQUARE expression RSQUARE'''
    v = p[1]
    if v == '(':
        p[0] = functionarguments(p[2])
    elif v == '[':
        p[0] = tsentry(p[2])