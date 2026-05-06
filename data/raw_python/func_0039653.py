def p_elseif_list(p):
    '''elseif_list : empty
                   | elseif_list ELSEIF LPAREN expr RPAREN statement'''
    if len(p) == 2:
        p[0] = []
    else:
        p[0] = p[1] + [ast.ElseIf(p[4], p[6], lineno=p.lineno(2))]