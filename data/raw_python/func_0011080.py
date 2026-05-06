def iscomplex(polynomial):
    """Returns whether the polynomial has complex coefficients

    :param polynomial: Polynomial of noncommutive variables.
    :type polynomial: :class:`sympy.core.expr.Expr`.

    :returns: bool -- whether there is a complex coefficient.
    """
    if isinstance(polynomial, (int, float)):
        return False
    if isinstance(polynomial, complex):
        return True
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, complex) or variable == I:
                return True
    return False