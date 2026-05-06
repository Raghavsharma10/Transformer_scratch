def _readedf_extractline(left, right):
    """Helper function to interpret lines in an EDF file header.
    """
    functions = [int, float, lambda l:float(l.split(None, 1)[0]),
                 lambda l:int(l.split(None, 1)[0]),
                 dateutil.parser.parse, lambda x:str(x)]
    for f in functions:
        try:
            right = f(right)
            break
        except ValueError:
            continue
    return right