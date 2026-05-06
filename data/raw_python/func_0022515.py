def solve_select(expr, vars):
    """Use IAssociative.select to get key (rhs) from the data (lhs).

    This operation supports both scalars and repeated values on the LHS -
    selecting from a repeated value implies a map-like operation and returns a
    new repeated value.
    """
    data, _ = __solve_for_repeated(expr.lhs, vars)
    key = solve(expr.rhs, vars).value

    try:
        results = [associative.select(d, key) for d in repeated.getvalues(data)]
    except (KeyError, AttributeError):
        # Raise a better exception for accessing a non-existent key.
        raise errors.EfilterKeyError(root=expr, key=key, query=expr.source)
    except (TypeError, ValueError):
        # Raise a better exception for what is probably a null pointer error.
        if vars.locals is None:
            raise errors.EfilterNoneError(
                root=expr, query=expr.source,
                message="Cannot select key %r from a null." % key)
        else:
            raise
    except NotImplementedError:
        raise errors.EfilterError(
            root=expr, query=expr.source,
            message="Cannot select keys from a non-associative value.")

    return Result(repeated.meld(*results), ())