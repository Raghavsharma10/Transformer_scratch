def _decorate_namespace_property(bases: List[type], namespace: MutableMapping[str, Any], key: str) -> None:
    """Collect contracts for all getters/setters/deleters corresponding to ``key`` and decorate them."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    value = namespace[key]
    assert isinstance(value, property)

    fget = value.fget  # type: Optional[Callable[..., Any]]
    fset = value.fset  # type: Optional[Callable[..., Any]]
    fdel = value.fdel  # type: Optional[Callable[..., Any]]

    for func in [value.fget, value.fset, value.fdel]:
        func = cast(Callable[..., Any], func)

        if func is None:
            continue

        # Collect the preconditions and postconditions from bases
        base_preconditions = []  # type: List[List[Contract]]
        base_snapshots = []  # type: List[Snapshot]
        base_postconditions = []  # type: List[Contract]

        bases_have_func = False
        for base in bases:
            if hasattr(base, key):
                base_property = getattr(base, key)
                assert isinstance(base_property, property), \
                    "Expected base {} to have {} as property, but got: {}".format(base, key, base_property)

                if func == value.fget:
                    base_func = getattr(base, key).fget
                elif func == value.fset:
                    base_func = getattr(base, key).fset
                elif func == value.fdel:
                    base_func = getattr(base, key).fdel
                else:
                    raise NotImplementedError("Unhandled case: func neither value.fget, value.fset nor value.fdel")

                if base_func is None:
                    continue

                bases_have_func = True

                # Check if there is a checker function in the base class
                base_contract_checker = icontract._checkers.find_checker(func=base_func)

                # Ignore functions which don't have preconditions or postconditions
                if base_contract_checker is not None:
                    base_preconditions.extend(base_contract_checker.__preconditions__)  # type: ignore
                    base_snapshots.extend(base_contract_checker.__postcondition_snapshots__)  # type: ignore
                    base_postconditions.extend(base_contract_checker.__postconditions__)  # type: ignore

        # Add preconditions and postconditions of the function
        preconditions = []  # type: List[List[Contract]]
        snapshots = []  # type: List[Snapshot]
        postconditions = []  # type: List[Contract]

        contract_checker = icontract._checkers.find_checker(func=func)
        if contract_checker is not None:
            preconditions = contract_checker.__preconditions__  # type: ignore
            snapshots = contract_checker.__postcondition_snapshots__
            postconditions = contract_checker.__postconditions__  # type: ignore

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
                if func == value.fget:
                    fget = contract_checker
                elif func == value.fset:
                    fset = contract_checker
                elif func == value.fdel:
                    fdel = contract_checker
                else:
                    raise NotImplementedError("Unhandled case: func neither fget, fset nor fdel")

            # Override the preconditions and postconditions
            contract_checker.__preconditions__ = preconditions  # type: ignore
            contract_checker.__postcondition_snapshots__ = snapshots  # type: ignore
            contract_checker.__postconditions__ = postconditions  # type: ignore

    if fget != value.fget or fset != value.fset or fdel != value.fdel:
        namespace[key] = property(fget=fget, fset=fset, fdel=fdel)