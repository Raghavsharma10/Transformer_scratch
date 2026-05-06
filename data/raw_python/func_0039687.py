def p_non_empty_array_pair_list_pair(p):
    '''non_empty_array_pair_list : non_empty_array_pair_list COMMA expr DOUBLE_ARROW AND variable
                                 | non_empty_array_pair_list COMMA expr DOUBLE_ARROW expr
                                 | expr DOUBLE_ARROW AND variable
                                 | expr DOUBLE_ARROW expr'''
    if len(p) == 7:
        p[0] = p[1] + [ast.ArrayElement(p[3], p[6], True, lineno=p.lineno(2))]
    elif len(p) == 6:
        p[0] = p[1] + [ast.ArrayElement(p[3], p[5], False, lineno=p.lineno(2))]
    elif len(p) == 5:
        p[0] = [ast.ArrayElement(p[1], p[4], True, lineno=p.lineno(2))]
    else:
        p[0] = [ast.ArrayElement(p[1], p[3], False, lineno=p.lineno(2))]