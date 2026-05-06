def _decorate_namespace_function(bases: List[type], namespace: MutableMapping[str, Any], key: str) -> None:
    """Collect preconditions and postconditions from the bases and decorate the function at the ``key``."""
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals

    value = namespace[key]
    assert inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod))

    # Determine the function to be decorated
    if inspect.isfunction(value):
        func = value
    elif isinstance(value, (staticmethod, classmethod)):
        func = value.__func__
    else:
        raise NotImplementedError("Unexpected value for a function: {}".format(value))

    # Collect preconditions and postconditions of the function
    preconditions = []  # type: List[List[Contract]]
    snapshots = []  # type: List[Snapshot]
    postconditions = []  # type: List[Contract]

    contract_checker = icontract._checkers.find_checker(func=func)
    if contract_checker is not None:
        preconditions = contract_checker.__preconditions__  # type: ignore
        snapshots = contract_checker.__postcondition_snapshots__  # type: ignore
        postconditions = contract_checker.__postconditions__  # type: ignore

    # Collect the preconditions and postconditions from bases.
    #
    # Preconditions and postconditions of __init__ of base classes are deliberately ignored (and not collapsed) since
    # initialization is an operation specific to the concrete class and does not relate to the class hierarchy.
    if key not in ['__init__']:
        base_preconditions = []  # type: List[List[Contract]]
        base_snapshots = []  # type: List[Snapshot]
        base_postconditions = []  # type: List[Contract]

        bases_have_func = False
        for base in bases:
            if hasattr(base, key):
                bases_have_func = True

                # Check if there is a checker function in the base class
                base_func = getattr(base, key)
                base_contract_checker = icontract._checkers.find_checker(func=base_func)

                # Ignore functions which don't have preconditions or postconditions
                if base_contract_checker is not None:
                    base_preconditions.extend(base_contract_checker.__preconditions__)  # type: ignore
                    base_snapshots.extend(base_contract_checker.__postcondition_snapshots__)  # type: ignore
                    base_postconditions.extend(base_contract_checker.__postconditions__)  # type: ignore

        # Collapse preconditions and postconditions from the bases with the the function's own ones
        preconditions = _collapse_preconditions(
            base_preconditions=base_preconditions,
            bases_have_func=bases_have_func,
            preconditions=preconditions,
            func=func)

        snapshots = _collapse_snapshots(base_snapshots=base_snapshots, snapshots=snapshots)

        postconditions = _collapse_postconditions(
            base_postconditions=base_postconditions, postconditions=postconditions)

    if preconditions or postconditions:
        if contract_checker is None:
            contract_checker = icontract._checkers.decorate_with_checker(func=func)

            # Replace the function with the function decorated with contract checks
            if inspect.isfunction(value):
                namespace[key] = contract_checker
            elif isinstance(value, staticmethod):
                namespace[key] = staticmethod(contract_checker)

            elif isinstance(value, classmethod):
                namespace[key] = classmethod(contract_checker)

            else:
                raise NotImplementedError("Unexpected value for a function: {}".format(value))

        # Override the preconditions and postconditions
        contract_checker.__preconditions__ = preconditions  # type: ignore
        contract_checker.__postcondition_snapshots__ = snapshots  # type: ignore
        contract_checker.__postconditions__ = postconditions