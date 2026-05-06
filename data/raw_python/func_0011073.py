def count_ncmonomials(monomials, degree):
    """Given a list of monomials, it counts those that have a certain degree,
    or less. The function is useful when certain monomials were eliminated
    from the basis.

    :param variables: The noncommutative variables making up the monomials
    :param monomials: List of monomials (the monomial basis).
    :param degree:  Maximum degree to count.

    :returns: The count of appropriate monomials.
    """
    ncmoncount = 0
    for monomial in monomials:
        if ncdegree(monomial) <= degree:
            ncmoncount += 1
        else:
            break
    return ncmoncount