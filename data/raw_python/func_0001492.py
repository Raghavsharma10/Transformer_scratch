def decorate_with_checker(func: CallableT) -> CallableT:
    """Decorate the function with a checker that verifies the preconditions and postconditions."""
    assert not hasattr(func, "__preconditions__"), \
        "Expected func to have no list of preconditions (there should be only a single contract checker per function)."

    assert not hasattr(func, "__postconditions__"), \
        "Expected func to have no list of postconditions (there should be only a single contract checker per function)."

    assert not hasattr(func, "__postcondition_snapshots__"), \
        "Expected func to have no list of postcondition snapshots (there should be only a single contract checker " \
        "per function)."

    sign = inspect.signature(func)
    param_names = list(sign.parameters.keys())

    # Determine the default argument values.
    kwdefaults = dict()  # type: Dict[str, Any]

    # Add to the defaults all the values that are needed by the contracts.
    for param in sign.parameters.values():
        if param.default != inspect.Parameter.empty:
            kwdefaults[param.name] = param.default

    def wrapper(*args, **kwargs):
        """Wrap func by checking the preconditions and postconditions."""
        preconditions = getattr(wrapper, "__preconditions__")  # type: List[List[Contract]]
        snapshots = getattr(wrapper, "__postcondition_snapshots__")  # type: List[Snapshot]
        postconditions = getattr(wrapper, "__postconditions__")  # type: List[Contract]

        resolved_kwargs = _kwargs_from_call(param_names=param_names, kwdefaults=kwdefaults, args=args, kwargs=kwargs)

        if postconditions:
            if 'result' in resolved_kwargs:
                raise TypeError("Unexpected argument 'result' in a function decorated with postconditions.")

            if 'OLD' in resolved_kwargs:
                raise TypeError("Unexpected argument 'OLD' in a function decorated with postconditions.")

        # Assert the preconditions in groups. This is necessary to implement "require else" logic when a class
        # weakens the preconditions of its base class.
        violation_err = None  # type: Optional[ViolationError]
        for group in preconditions:
            violation_err = None
            try:
                for contract in group:
                    _assert_precondition(contract=contract, resolved_kwargs=resolved_kwargs)
                break
            except ViolationError as err:
                violation_err = err

        if violation_err is not None:
            raise violation_err  # pylint: disable=raising-bad-type

        # Capture the snapshots
        if postconditions:
            old_as_mapping = dict()  # type: MutableMapping[str, Any]
            for snap in snapshots:
                # This assert is just a last defense.
                # Conflicting snapshot names should have been caught before, either during the decoration or
                # in the meta-class.
                assert snap.name not in old_as_mapping, "Snapshots with the conflicting name: {}"
                old_as_mapping[snap.name] = _capture_snapshot(a_snapshot=snap, resolved_kwargs=resolved_kwargs)

            resolved_kwargs['OLD'] = _Old(mapping=old_as_mapping)

        # Execute the wrapped function
        result = func(*args, **kwargs)

        if postconditions:
            resolved_kwargs['result'] = result

            # Assert the postconditions as a conjunction
            for contract in postconditions:
                _assert_postcondition(contract=contract, resolved_kwargs=resolved_kwargs)

        return result  # type: ignore

    # Copy __doc__ and other properties so that doctests can run
    functools.update_wrapper(wrapper=wrapper, wrapped=func)

    assert not hasattr(wrapper, "__preconditions__"), "Expected no preconditions set on a pristine contract checker."
    assert not hasattr(wrapper, "__postcondition_snapshots__"), \
        "Expected no postcondition snapshots set on a pristine contract checker."
    assert not hasattr(wrapper, "__postconditions__"), "Expected no postconditions set on a pristine contract checker."

    # Precondition is a list of condition groups (i.e. disjunctive normal form):
    # each group consists of AND'ed preconditions, while the groups are OR'ed.
    #
    # This is necessary in order to implement "require else" logic when a class weakens the preconditions of
    # its base class.
    setattr(wrapper, "__preconditions__", [])
    setattr(wrapper, "__postcondition_snapshots__", [])
    setattr(wrapper, "__postconditions__", [])

    return wrapper