def get_one_over_n_factorial(counter_entry):
    r"""
    Calculates  the :math:`\frac{1}{\mathbf{n!}}` of eq. 6 (see Ale et al. 2013).
    That is the invert of a product of factorials.
    :param counter_entry: an entry of counter. That is an array of integers of length equal to the number of variables.
    For instance, `counter_entry` could be `[1,0,1]` for three variables.
    :return: a scalar as a sympy expression
    """
    # compute all factorials
    factos = [sp.factorial(c) for c in counter_entry]
    # multiply them
    prod = product(factos)
    # return the invert
    return sp.Integer(1)/sp.S(prod)