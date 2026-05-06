def pick_monomials_of_degree(monomials, degree):
    """Collect all monomials up of a given degree.
    """
    selected_monomials = []
    for monomial in monomials:
        if ncdegree(monomial) == degree:
            selected_monomials.append(monomial)
    return selected_monomials