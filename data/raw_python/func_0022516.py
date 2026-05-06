def solve_resolve(expr, vars):
    """Use IStructured.resolve to get member (rhs) from the object (lhs).

    This operation supports both scalars and repeated values on the LHS -
    resolving from a repeated value implies a map-like operation and returns a
    new repeated values.
    """
    objs, _ = __solve_for_repeated(expr.lhs, vars)
    member = solve(expr.rhs, vars).value

    try:
        results = [structured.resolve(o, member)
                   for o in repeated.getvalues(objs)]
    except (KeyError, AttributeError):
        # Raise a better exception for the non-existent member.
        raise errors.EfilterKeyError(root=expr.rhs, key=member,
                                     query=expr.source)
    except (TypeError, ValueError):
        # Is this a null object error?
        if vars.locals is None:
            raise errors.EfilterNoneError(
                root=expr, query=expr.source,
                message="Cannot resolve member %r from a null." % member)
        else:
            raise
    except NotImplementedError:
        raise errors.EfilterError(
            root=expr, query=expr.source,
            message="Cannot resolve members from a non-structured value.")

    return Result(repeated.meld(*results), ())