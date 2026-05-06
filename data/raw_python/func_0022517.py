def solve_apply(expr, vars):
    """Returns the result of applying function (lhs) to its arguments (rest).

    We use IApplicative to apply the function, because that gives the host
    application an opportunity to compare the function being called against
    a whitelist. EFILTER will never directly call a function that wasn't
    provided through a protocol implementation.
    """
    func = __solve_for_scalar(expr.func, vars)
    args = []
    kwargs = {}
    for arg in expr.args:
        if isinstance(arg, ast.Pair):
            if not isinstance(arg.lhs, ast.Var):
                raise errors.EfilterError(
                    root=arg.lhs,
                    message="Invalid argument name.")

            kwargs[arg.key.value] = solve(arg.value, vars).value
        else:
            args.append(solve(arg, vars).value)

    result = applicative.apply(func, args, kwargs)

    return Result(result, ())