def get_all_monomials(variables, extramonomials, substitutions, degree,
                      removesubstitutions=True):
    """Return the monomials of a certain degree.
    """
    monomials = get_monomials(variables, degree)
    if extramonomials is not None:
        monomials.extend(extramonomials)
    if removesubstitutions and substitutions is not None:
        monomials = [monomial for monomial in monomials if monomial not
                     in substitutions]
        monomials = [remove_scalar_factor(apply_substitutions(monomial,
                                                              substitutions))
                     for monomial in monomials]
    monomials = unique(monomials)
    return monomials