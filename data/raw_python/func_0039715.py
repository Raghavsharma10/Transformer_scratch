def p_static_non_empty_array_pair_list_item(p):
    '''static_non_empty_array_pair_list : static_non_empty_array_pair_list COMMA static_scalar
                                        | static_scalar'''
    if len(p) == 4:
        p[0] = p[1] + [ast.ArrayElement(None, p[3], False, lineno=p.lineno(2))]
    else:
        p[0] = [ast.ArrayElement(None, p[1], False, lineno=p.lineno(1))]