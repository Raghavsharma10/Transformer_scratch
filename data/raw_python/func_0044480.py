def p_query(p):
    """
    query : QUERY IDENTIFIER SLASH IDENTIFIER
          | QUERY IDENTIFIER SLASH IDENTIFIER OPEN_CURLY_BRACKET query_option_list CLOSE_CURLY_BRACKET
    """
    parameters = {"service": p[2], "operation": p[4]}
    if len(p) > 5:
        parameters = merge_map(parameters, p[6])
    p[0] = Query(**parameters)