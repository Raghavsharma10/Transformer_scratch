def diffPrefsPrior(priorstring):
    """Parses `priorstring` and returns `prior` tuple."""
    assert isinstance(priorstring, str)
    prior = priorstring.split(',')
    if len(prior) == 3 and prior[0] == 'invquadratic':
        [c1, c2] = [float(x) for x in prior[1 : ]]
        assert c1 > 0 and c2 > 0, "C1 and C2 must be > 1 for invquadratic prior"
        return ('invquadratic', c1, c2)
    else:
        raise ValueError("Invalid diffprefsprior: {0}".format(priorstring))