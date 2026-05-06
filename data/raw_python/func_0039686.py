def p_non_empty_array_pair_list_item(p):
    '''non_empty_array_pair_list : non_empty_array_pair_list COMMA AND variable
                                 | non_empty_array_pair_list COMMA expr
                                 | AND variable
                                 | expr'''
    if len(p) == 5:
        p[0] = p[1] + [ast.ArrayElement(None, p[4], True, lineno=p.lineno(2))]
    elif len(p) == 4:
        p[0] = p[1] + [ast.ArrayElement(None, p[3], False, lineno=p.lineno(2))]
    elif len(p) == 3:
        p[0] = [ast.ArrayElement(None, p[2], True, lineno=p.lineno(1))]
    else:
        p[0] = [ast.ArrayElement(None, p[1], False, lineno=p.lineno(1))]