def apply_on_each_func_args_sig(func,
                                cur_args,
                                cur_kwargs,
                                sig,  # type: Signature
                                func_to_apply,
                                func_to_apply_params_dict):
    """
    Applies func_to_apply on each argument of func according to what's received in current call (cur_args, cur_kwargs).
    For each argument of func named 'att' in its signature, the following method is called:

    `func_to_apply(cur_att_value, func_to_apply_paramers_dict[att], func, att_name)`

    :param func:
    :param cur_args:
    :param cur_kwargs:
    :param sig:
    :param func_to_apply:
    :param func_to_apply_params_dict:
    :return:
    """

    # match the received arguments with the signature to know who is who
    bound_values = sig.bind(*cur_args, **cur_kwargs)

    # add the default values in here to get a full list
    apply_defaults(bound_values)

    for att_name, att_value in bound_values.arguments.items():
        if att_name in func_to_apply_params_dict.keys():
            # value = a normal value, or cur_kwargs as a whole
            func_to_apply(att_value, func_to_apply_params_dict[att_name], func, att_name)