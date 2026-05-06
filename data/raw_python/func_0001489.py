def _assert_precondition(contract: Contract, resolved_kwargs: Mapping[str, Any]) -> None:
    """
    Assert that the contract holds as a precondition.

    :param contract: contract to be verified
    :param resolved_kwargs: resolved keyword arguments (including the default values)
    :return:
    """
    # Check that all arguments to the condition function have been set.
    missing_args = [arg_name for arg_name in contract.condition_args if arg_name not in resolved_kwargs]
    if missing_args:
        raise TypeError(
            ("The argument(s) of the precondition have not been set: {}. "
             "Does the original function define them? Did you supply them in the call?").format(missing_args))

    condition_kwargs = {
        arg_name: value
        for arg_name, value in resolved_kwargs.items() if arg_name in contract.condition_arg_set
    }

    check = contract.condition(**condition_kwargs)

    if not check:
        if contract.error is not None and (inspect.ismethod(contract.error) or inspect.isfunction(contract.error)):
            assert contract.error_arg_set is not None, "Expected error_arg_set non-None if contract.error a function."
            assert contract.error_args is not None, "Expected error_args non-None if contract.error a function."

            error_kwargs = {
                arg_name: value
                for arg_name, value in resolved_kwargs.items() if arg_name in contract.error_arg_set
            }

            missing_args = [arg_name for arg_name in contract.error_args if arg_name not in resolved_kwargs]
            if missing_args:
                msg_parts = []  # type: List[str]
                if contract.location is not None:
                    msg_parts.append("{}:\n".format(contract.location))

                msg_parts.append(
                    ("The argument(s) of the precondition error have not been set: {}. "
                     "Does the original function define them? Did you supply them in the call?").format(missing_args))

                raise TypeError(''.join(msg_parts))

            raise contract.error(**error_kwargs)

        else:
            msg = icontract._represent.generate_message(contract=contract, condition_kwargs=condition_kwargs)
            if contract.error is None:
                raise ViolationError(msg)
            elif isinstance(contract.error, type):
                raise contract.error(msg)