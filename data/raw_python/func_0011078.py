def get_monomials(variables, degree):
    """Generates all noncommutative monomials up to a degree

    :param variables: The noncommutative variables to generate monomials from
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param degree: The maximum degree.
    :type degree: int.

    :returns: list of monomials.
    """
    if degree == -1:
        return []
    if not variables:
        return [S.One]
    else:
        _variables = variables[:]
        _variables.insert(0, 1)
        ncmonomials = [S.One]
        ncmonomials.extend(var for var in variables)
        for var in variables:
            if not is_hermitian(var):
                ncmonomials.append(var.adjoint())
        for _ in range(1, degree):
            temp = []
            for var in _variables:
                for new_var in ncmonomials:
                    temp.append(var * new_var)
                    if var != 1 and not is_hermitian(var):
                        temp.append(var.adjoint() * new_var)
            ncmonomials = unique(temp[:])
        return ncmonomials