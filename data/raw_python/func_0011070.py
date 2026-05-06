def get_support(variables, polynomial):
    """Gets the support of a polynomial.
    """
    support = []
    if is_number_type(polynomial):
        support.append([0] * len(variables))
        return support
    for monomial in polynomial.expand().as_coefficients_dict():
        tmp_support = [0] * len(variables)
        mon, _ = __separate_scalar_factor(monomial)
        symbolic_support = flatten(split_commutative_parts(mon))
        for s in symbolic_support:
            if isinstance(s, Pow):
                base = s.base
                if is_adjoint(base):
                    base = base.adjoint()
                tmp_support[variables.index(base)] = s.exp
            elif is_adjoint(s):
                tmp_support[variables.index(s.adjoint())] = 1
            elif isinstance(s, (Operator, Symbol)):
                tmp_support[variables.index(s)] = 1
        support.append(tmp_support)
    return support