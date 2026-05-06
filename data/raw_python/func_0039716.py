def p_static_non_empty_array_pair_list_pair(p):
    '''static_non_empty_array_pair_list : static_non_empty_array_pair_list COMMA static_scalar DOUBLE_ARROW static_scalar
                                        | static_scalar DOUBLE_ARROW static_scalar'''
    if len(p) == 6:
        p[0] = p[1] + [ast.ArrayElement(p[3], p[5], False, lineno=p.lineno(2))]
    else:
        p[0] = [ast.ArrayElement(p[1], p[3], False, lineno=p.lineno(2))]