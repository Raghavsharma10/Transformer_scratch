def p_define_service(p):
    """
    define_service : SERVICE IDENTIFIER OPEN_CURLY_BRACKET settings operation_list CLOSE_CURLY_BRACKET
                   | SERVICE IDENTIFIER OPEN_CURLY_BRACKET operation_list CLOSE_CURLY_BRACKET
    """
    if len(p) == 7:
        body = p[4] + p[5]
    else:
        body = p[4]
    p[0] = DefineService(p[2], body)