def solve_map(expr, vars):
    """Solves the map-form, by recursively calling its RHS with new vars.

    let-forms are binary expressions. The LHS should evaluate to an IAssociative
    that can be used as new vars with which to solve a new query, of which
    the RHS is the root. In most cases, the LHS will be a Var (var).

    Typically, map-forms result from the dotty "dot" (.) operator. For example,
    the query "User.name" will translate to a map-form with the var "User"
    on LHS and a var to "name" on the RHS. With top-level vars being
    something like {"User": {"name": "Bob"}}, the Var on the LHS will
    evaluate to {"name": "Bob"}, which subdict will then be used on the RHS as
    new vars, and that whole form will evaluate to "Bob".
    """
    lhs_values, _ = __solve_for_repeated(expr.lhs, vars)

    def lazy_map():
        try:
            for lhs_value in repeated.getvalues(lhs_values):
                yield solve(expr.rhs,
                            __nest_scope(expr.lhs, vars, lhs_value)).value
        except errors.EfilterNoneError as error:
            error.root = expr
            raise

    return Result(repeated.lazy(lazy_map), ())