def p_statement_if(p):
    '''statement : IF LPAREN expr RPAREN statement elseif_list else_single
                 | IF LPAREN expr RPAREN COLON inner_statement_list new_elseif_list new_else_single ENDIF SEMI'''
    if len(p) == 8:
        p[0] = ast.If(p[3], p[5], p[6], p[7], lineno=p.lineno(1))
    else:
        p[0] = ast.If(p[3], ast.Block(p[6], lineno=p.lineno(5)),
                      p[7], p[8], lineno=p.lineno(1))