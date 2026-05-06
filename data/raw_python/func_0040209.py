def is_subsumed_by(x, y):
    """
    Returns true if y subsumes x (for example P(x) subsumes P(A) as it is more
    abstract)
    """
    varsX = __split_expression(x)[1]
    theta = unify(x, y)
    if theta is problem.FAILURE:
        return False
    return all(__is_variable(theta[var]) for var in theta.keys()
               if var in varsX)