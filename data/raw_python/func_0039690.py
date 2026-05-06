def p_lexical_var_list(p):
    '''lexical_var_list : lexical_var_list COMMA AND VARIABLE
                        | lexical_var_list COMMA VARIABLE
                        | AND VARIABLE
                        | VARIABLE'''
    if len(p) == 5:
        p[0] = p[1] + [ast.LexicalVariable(p[4], True, lineno=p.lineno(2))]
    elif len(p) == 4:
        p[0] = p[1] + [ast.LexicalVariable(p[3], False, lineno=p.lineno(2))]
    elif len(p) == 3:
        p[0] = [ast.LexicalVariable(p[2], True, lineno=p.lineno(1))]
    else:
        p[0] = [ast.LexicalVariable(p[1], False, lineno=p.lineno(1))]