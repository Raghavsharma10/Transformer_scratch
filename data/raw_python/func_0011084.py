def save_monomial_index(filename, monomial_index):
    """Save a monomial dictionary for debugging purposes.

    :param filename: The name of the file to save to.
    :type filename: str.
    :param monomial_index: The monomial index of the SDP relaxation.
    :type monomial_index: dict of :class:`sympy.core.expr.Expr`.

    """
    monomial_translation = [''] * (len(monomial_index) + 1)
    for key, k in monomial_index.items():
        monomial_translation[k] = convert_monomial_to_string(key)
    file_ = open(filename, 'w')
    for k in range(len(monomial_translation)):
        file_.write('%s %s\n' % (k, monomial_translation[k]))
    file_.close()