def simplify_polynomial(polynomial, monomial_substitutions):
    """Simplify a polynomial for uniform handling later.
    """
    if isinstance(polynomial, (int, float, complex)):
        return polynomial
    polynomial = (1.0 * polynomial).expand(mul=True,
                                           multinomial=True)
    if is_number_type(polynomial):
        return polynomial
    if polynomial.is_Mul:
        elements = [polynomial]
    else:
        elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
    new_polynomial = 0
    # Identify its constituent monomials
    for element in elements:
        monomial, coeff = separate_scalar_factor(element)
        monomial = apply_substitutions(monomial, monomial_substitutions)
        new_polynomial += coeff * monomial
    return new_polynomial