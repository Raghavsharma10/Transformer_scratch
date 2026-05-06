def p_fail(p):
    """
    fail : FAIL NUMBER
         | FAIL
    """
    if len(p) > 2:
        p[0] = Fail(float(p[2]))
    else:
        p[0] = Fail()