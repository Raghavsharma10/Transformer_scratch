def __separate_scalar_factor(monomial):
    """Separate the constant factor from a monomial.
    """
    scalar_factor = 1
    if is_number_type(monomial):
        return S.One, monomial
    if monomial == 0:
        return S.One, 0
    comm_factors, _ = split_commutative_parts(monomial)
    if len(comm_factors) > 0:
        if isinstance(comm_factors[0], Number):
            scalar_factor = comm_factors[0]
    if scalar_factor != 1:
        return monomial / scalar_factor, scalar_factor
    else:
        return monomial, scalar_factor