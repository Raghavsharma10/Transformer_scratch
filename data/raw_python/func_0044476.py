def p_autoscaling_setting(p):
    """
    autoscaling_setting : PERIOD COLON NUMBER
                        | LIMITS COLON OPEN_SQUARE_BRACKET NUMBER COMMA NUMBER CLOSE_SQUARE_BRACKET
    """
    if len(p) == 8:
        p[0] = {"limits": (int(p[4]), int(p[6]))}
    elif len(p) == 4:
        p[0] = {"period": int(p[3])}
    else:
        raise RuntimeError("Invalid product in 'autoscaling_setting'")