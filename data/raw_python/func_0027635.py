def decorate_with_validators(func,
                             func_signature=None,  # type: Signature
                             **validators          # type: Validator
                             ):
    """
    Utility method to decorate the provided function with the provided input and output Validator objects. Since this
    method takes Validator objects as argument, it is for advanced users.

    :param func: the function to decorate. It might already be decorated, this method will check it and wont create
    another wrapper in this case, simply adding the validators to the existing wrapper
    :param func_signature: the function's signature if it is already known (internal calls), otherwise it will be found
    again by inspection
    :param validators: a dictionary of arg_name (or _out_) => Validator or list of Validator
    :return:
    """
    # first turn the dictionary values into lists only
    for arg_name, validator in validators.items():
        if not isinstance(validator, list):
            validators[arg_name] = [validator]

    if hasattr(func, '__wrapped__') and hasattr(func.__wrapped__, '__validators__'):
        # ---- This function is already wrapped by our validation wrapper ----

        # Update the dictionary of validators with the new validator(s)
        for arg_name, validator in validators.items():
            for v in validator:
                if arg_name in func.__wrapped__.__validators__:
                    func.__wrapped__.__validators__[arg_name].append(v)
                else:
                    func.__wrapped__.__validators__[arg_name] = [v]

        # return the function, no need to wrap it further (it is already wrapped)
        return func

    else:
        # ---- This function is not yet wrapped by our validator. ----

        # Store the dictionary of validators as an attribute of the function
        if hasattr(func, '__validators__'):
            raise ValueError('Function ' + str(func) + ' already has a defined __validators__ attribute, valid8 '
                             'decorators can not be applied on it')
        else:
            try:
                func.__validators__ = validators
            except AttributeError:
                raise ValueError("Error - Could not add validators list to function '%s'" % func)

        # either reuse or recompute function signature
        func_signature = func_signature or signature(func)

        # create a wrapper with the same signature
        @wraps(func)
        def validating_wrapper(*args, **kwargs):
            """ This is the wrapper that will be called everytime the function is called """

            # (a) Perform input validation by applying `_assert_input_is_valid` on all received arguments
            apply_on_each_func_args_sig(func, args, kwargs, func_signature,
                                        func_to_apply=_assert_input_is_valid,
                                        func_to_apply_params_dict=func.__validators__)

            # (b) execute the function as usual
            res = func(*args, **kwargs)

            # (c) validate output if needed
            if _OUT_KEY in func.__validators__:
                for validator in func.__validators__[_OUT_KEY]:
                    validator.assert_valid(res)

            return res

        return validating_wrapper