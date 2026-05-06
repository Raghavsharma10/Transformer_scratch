def p_new_elseif_list(p):
    '''new_elseif_list : empty
                       | new_elseif_list ELSEIF LPAREN expr RPAREN COLON inner_statement_list'''
    if len(p) == 2:
        p[0] = []
    else:
        p[0] = p[1] + [ast.ElseIf(p[4], ast.Block(p[7], lineo=p.lineno(6)),
                                  lineno=p.lineno(2))]