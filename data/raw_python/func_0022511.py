def __solve_for_repeated(expr, vars):
    """Helper: solve 'expr' always returning an IRepeated.

    If the result of solving 'expr' is a list or a tuple of IStructured objects
    then treat is as a repeated value of IStructured objects because that's
    what the called meant to do. This is a convenience helper so users of the
    API don't have to create IRepeated objects.

    If the result of solving 'expr' is a scalar then return it as a repeated
    value of one element.

    Arguments:
        expr: Expression to solve.
        vars: The scope.

    Returns:
        IRepeated result of solving 'expr'.
        A booelan to indicate whether the original was repeating.
    """
    var = solve(expr, vars).value
    if (var and isinstance(var, (tuple, list))
            and protocol.implements(var[0], structured.IStructured)):
        return repeated.meld(*var), False

    return var, repeated.isrepeating(var)