def p_query_option_list(p):
    """
    query_option_list : query_option COMMA query_option_list
                      | query_option
    """
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = merge_map(p[1], p[3])
    else:
        raise RuntimeError("Invalid product rules for 'query_option_list'")