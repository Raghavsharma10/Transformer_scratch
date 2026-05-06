def p_arguments(p):
    """arguments : argumentlist
                 | argumentlist test
                 | argumentlist '(' testlist ')'"""
    p[0] = { 'args' : p[1], }
    if len(p) > 2:
        if p[2] == '(':
            p[0]['tests'] = p[3]
        else:
            p[0]['tests'] = [ p[2] ]