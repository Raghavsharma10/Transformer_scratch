def tree(X, n, alpha=0, symbolic=False):
    """Recurrence coefficients for generalized Laguerre polynomials. Set
    alpha=0 (default) to get classical Laguerre.
    """
    args = recurrence_coefficients(n, alpha=alpha, symbolic=symbolic)
    return line_tree(X, *args)