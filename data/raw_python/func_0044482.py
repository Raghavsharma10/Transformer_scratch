def p_invoke(p):
    """
    invoke : INVOKE IDENTIFIER SLASH IDENTIFIER
           | INVOKE IDENTIFIER SLASH IDENTIFIER OPEN_CURLY_BRACKET PRIORITY COLON NUMBER CLOSE_CURLY_BRACKET
    """
    priority = None
    if len(p) > 5:
        priority = int(p[8])
    p[0] = Trigger(p[2], p[4], priority)