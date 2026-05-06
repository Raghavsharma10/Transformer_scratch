def _decorate_with_invariants(func: CallableT, is_init: bool) -> CallableT:
    """
    Decorate the function ``func`` of the class ``cls`` with invariant checks.

    If the function has been already decorated with invariant checks, the function returns immediately.

    :param func: function to be wrapped
    :param is_init: True if the ``func`` is __init__
    :return: function wrapped with invariant checks
    """
    if _already_decorated_with_invariants(func=func):
        return func

    sign = inspect.signature(func)
    param_names = list(sign.parameters.keys())

    if is_init:

        def wrapper(*args, **kwargs):
            """Wrap __init__ method of a class by checking the invariants *after* the invocation."""
            result = func(*args, **kwargs)
            instance = _find_self(param_names=param_names, args=args, kwargs=kwargs)

            for contract in instance.__class__.__invariants__:
                _assert_invariant(contract=contract, instance=instance)

            return result
    else:

        def wrapper(*args, **kwargs):
            """Wrap a function of a class by checking the invariants *before* and *after* the invocation."""
            instance = _find_self(param_names=param_names, args=args, kwargs=kwargs)

            for contract in instance.__class__.__invariants__:
                _assert_invariant(contract=contract, instance=instance)

            result = func(*args, **kwargs)

            for contract in instance.__class__.__invariants__:
                _assert_invariant(contract=contract, instance=instance)

            return result

    functools.update_wrapper(wrapper=wrapper, wrapped=func)

    setattr(wrapper, "__is_invariant_check__", True)

    return wrapper