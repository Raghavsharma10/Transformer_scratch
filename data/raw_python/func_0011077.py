def generate_operators(name, n_vars=1, hermitian=None, commutative=False):
    """Generates a number of commutative or noncommutative operators

    :param name: The prefix in the symbolic representation of the noncommuting
                 variables. This will be suffixed by a number from 0 to
                 n_vars-1 if n_vars > 1.
    :type name: str.
    :param n_vars: The number of variables.
    :type n_vars: int.
    :param hermitian: Optional parameter to request Hermitian variables .
    :type hermitian: bool.
    :param commutative: Optional parameter to request commutative variables.
                        Commutative variables are Hermitian by default.
    :type commutative: bool.

    :returns: list of :class:`sympy.physics.quantum.operator.Operator` or
              :class:`sympy.physics.quantum.operator.HermitianOperator`
              variables

    :Example:

    >>> generate_variables('y', 2, commutative=True)
    ￼[y0, y1]
    """

    variables = []
    for i in range(n_vars):
        if n_vars > 1:
            var_name = '%s%s' % (name, i)
        else:
            var_name = '%s' % name
        if hermitian is not None and hermitian:
            variables.append(HermitianOperator(var_name))
        else:
            variables.append(Operator(var_name))
        variables[-1].is_commutative = commutative
    return variables