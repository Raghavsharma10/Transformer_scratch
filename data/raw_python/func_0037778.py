def suppress_output(reverse=False):
    """
    Suppress output
    """
    if reverse:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    else:
        sys.stdout = os.devnull
        sys.stderr = os.devnull