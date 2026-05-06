def ncdegree(polynomial):
    """Returns the degree of a noncommutative polynomial.

    :param polynomial: Polynomial of noncommutive variables.
    :type polynomial: :class:`sympy.core.expr.Expr`.

    :returns: int -- the degree of the polynomial.
    """
    degree = 0
    if is_number_type(polynomial):
        return degree
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        subdegree = 0
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, Pow):
                subdegree += variable.exp
            elif not isinstance(variable, Number) and variable != I:
                subdegree += 1
        if subdegree > degree:
            degree = subdegree
    return degree