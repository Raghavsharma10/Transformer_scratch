def constant(constanttype):
    """
    Get A Constant
    """
    constanttype = constanttype.lower()
    if constanttype == 'pi':
        return math.pi
    elif constanttype == 'e':
        return math.e
    elif constanttype == 'tau':
        return math.tau
    elif constanttype == 'inf':
        return math.inf
    elif constanttype == 'nan':
        return math.nan
    elif constanttype in ['phi', 'golden']:
        return (1 + 5**0.5) / 2