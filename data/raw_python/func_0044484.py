def p_retry_option_list(p):
    """
    retry_option_list : retry_option COMMA retry_option_list
                      | retry_option
    """
    if len(p) == 4:
        p[0] = merge_map(p[1], p[3])
    elif len(p) == 2:
        p[0] = p[1]
    else:
        raise RuntimeError("Invalid production in 'retry_option_list'")