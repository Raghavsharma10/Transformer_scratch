def solve_var(expr, vars):
    """Returns the value of the var named in the expression."""
    try:
        return Result(structured.resolve(vars, expr.value), ())
    except (KeyError, AttributeError) as e:
        # Raise a better exception for accessing a non-existent member.
        raise errors.EfilterKeyError(root=expr, key=expr.value, message=e,
                                     query=expr.source)
    except (TypeError, ValueError) as e:
        # Raise a better exception for what is probably a null pointer error.
        if vars.locals is None:
            raise errors.EfilterNoneError(
                root=expr, query=expr.source,
                message="Trying to access member %r of a null." % expr.value)
        else:
            raise errors.EfilterTypeError(
                root=expr, query=expr.source,
                message="%r (vars: %r)" % (e, vars))
    except NotImplementedError as e:
        raise errors.EfilterError(
            root=expr, query=expr.source,
            message="Trying to access member %r of an instance of %r." %
            (expr.value, type(vars)))