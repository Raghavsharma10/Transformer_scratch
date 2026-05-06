def dprintx(passeditem, special=False):
    """Print Text if DEBUGALL set, optionally with PrettyPrint.

    Args:
        passeditem (str): item to print
        special (bool): determines if item prints with PrettyPrint
                        or regular print.

    """
    if DEBUGALL:
        if special:
            from pprint import pprint
            pprint(passeditem)
        else:
            print("%s%s%s" % (C_TI, passeditem, C_NORM))