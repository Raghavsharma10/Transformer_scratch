def p_throttling(p):
    """
    throttling : THROTTLING COLON NONE
               | THROTTLING COLON TAIL_DROP OPEN_BRACKET NUMBER CLOSE_BRACKET
    """
    throttling = NoThrottlingSettings()
    if len(p) == 7:
        throttling = TailDropSettings(int(p[5]))
    p[0] = {"throttling": throttling}