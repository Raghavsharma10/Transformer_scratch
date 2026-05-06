def derive_expr_from_counter_entry(expression, species, counter_entry):
    r"""
    Derives an given expression with respect to arbitrary species and orders.
    This is used to compute :math:`\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}}` in eq. 6

    :param expression: the expression to be derived
    :type expression: :class:`~sympy.Expr`
    :param species: the name of the variables (typically {y_0, y_1, ..., y_n})
    :type species: list[:class:`~sympy.Symbol`]
    :param counter_entry: an entry of counter. That is a tuple of integers of length equal to the number of variables.
    For example, (0,2,1) means we derive with respect to the third variable (first order)
    and to the second variable (second order)

    :return: the derived expression
    """

    # no derivation, we return the unchanged expression
    if sum(counter_entry) == 0:
        return expression

    # repeat a variable as many time as its value in counter
    diff_vars = reduce(operator.add, map(lambda v, c: [v] * c, species, counter_entry))
    out_expr = expression

    for var in diff_vars:
        # If the derivative is already 0, we can return 0
        if out_expr.is_Integer:
            return sp.Integer(0)
        out_expr = _cached_diff(out_expr, var)

    return out_expr