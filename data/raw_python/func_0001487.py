def find_checker(func: CallableT) -> Optional[CallableT]:
    """Iterate through the decorator stack till we find the contract checker."""
    contract_checker = None  # type: Optional[CallableT]
    for a_wrapper in _walk_decorator_stack(func):
        if hasattr(a_wrapper, "__preconditions__") or hasattr(a_wrapper, "__postconditions__"):
            contract_checker = a_wrapper

    return contract_checker