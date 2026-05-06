def p_retry_option(p):
    """
    retry_option : LIMIT COLON NUMBER
                 | DELAY COLON IDENTIFIER OPEN_BRACKET NUMBER CLOSE_BRACKET
    """
    if len(p) == 4:
        p[0] = {"limit": int(p[3]) }
    elif len(p) == 7:
        p[0] = {"delay": Delay(int(p[5]), p[3])}
    else:
        raise RuntimeError("Invalid production in 'retry_option'")