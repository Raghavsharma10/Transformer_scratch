def separate_scalar_factor(element):
    """Construct a monomial with the coefficient separated
    from an element in a polynomial.
    """
    coeff = 1.0
    monomial = S.One
    if isinstance(element, (int, float, complex)):
        coeff *= element
        return monomial, coeff
    for var in element.as_coeff_mul()[1]:
        if not (var.is_Number or var.is_imaginary):
            monomial = monomial * var
        else:
            if var.is_Number:
                coeff = float(var)
            # If not, then it is imaginary
            else:
                coeff = 1j * coeff
    coeff = float(element.as_coeff_mul()[0]) * coeff
    return monomial, coeff