def get_support_variables(polynomial):
    """Gets the support of a polynomial.
    """
    support = []
    if is_number_type(polynomial):
        return support
    for monomial in polynomial.expand().as_coefficients_dict():
        mon, _ = __separate_scalar_factor(monomial)
        symbolic_support = flatten(split_commutative_parts(mon))
        for s in symbolic_support:
            if isinstance(s, Pow):
                base = s.base
                if is_adjoint(base):
                    base = base.adjoint()
                support.append(base)
            elif is_adjoint(s):
                support.append(s.adjoint())
            elif isinstance(s, Operator):
                support.append(s)
    return support