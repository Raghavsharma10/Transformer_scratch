def printmp(msg):
    """Print temporarily, until next print overrides it.
    """
    filler = (80 - len(msg)) * ' '
    print(msg + filler, end='\r')
    sys.stdout.flush()