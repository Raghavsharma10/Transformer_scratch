def _already_decorated_with_invariants(func: CallableT) -> bool:
    """Check if the function has been already decorated with an invariant check by going through its decorator stack."""
    already_decorated = False
    for a_decorator in _walk_decorator_stack(func=func):
        if getattr(a_decorator, "__is_invariant_check__", False):
            already_decorated = True
            break

    return already_decorated