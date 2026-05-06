def _assert_input_is_valid(input_value,     # type: Any
                           validators,      # type: List[InputValidator]
                           validated_func,  # type: Callable
                           input_name       # type: str
                           ):
    """
    Called by the `validating_wrapper` in the first step (a) `apply_on_each_func_args` for each function input before
    executing the function. It simply delegates to the validator. The signature of this function is hardcoded to
    correspond to `apply_on_each_func_args`'s behaviour and should therefore not be changed.

    :param input_value: the value to validate
    :param validator: the Validator object that will be applied on input_value_to_validate
    :param validated_func: the function for which this validation is performed. This is not used since the Validator
        knows it already, but we should not change the signature here.
    :param input_name: the name of the function input that is being validated
    :return: Nothing
    """
    for validator in validators:
        validator.assert_valid(input_name, input_value)