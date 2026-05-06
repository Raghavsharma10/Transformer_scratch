def _assert_invariant(contract: Contract, instance: Any) -> None:
    """Assert that the contract holds as a class invariant given the instance of the class."""
    if 'self' in contract.condition_arg_set:
        check = contract.condition(self=instance)
    else:
        check = contract.condition()

    if not check:
        if contract.error is not None and (inspect.ismethod(contract.error) or inspect.isfunction(contract.error)):
            assert contract.error_arg_set is not None, "Expected error_arg_set non-None if contract.error a function."
            assert contract.error_args is not None, "Expected error_args non-None if contract.error a function."

            if 'self' in contract.error_arg_set:
                raise contract.error(self=instance)
            else:
                raise contract.error()
        else:
            if 'self' in contract.condition_arg_set:
                msg = icontract._represent.generate_message(contract=contract, condition_kwargs={"self": instance})
            else:
                msg = icontract._represent.generate_message(contract=contract, condition_kwargs=dict())

            if contract.error is None:
                raise ViolationError(msg)
            elif isinstance(contract.error, type):
                raise contract.error(msg)
            else:
                raise NotImplementedError("Unhandled contract.error: {}".format(contract.error))