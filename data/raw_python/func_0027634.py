def _get_final_none_policy_for_validator(is_nonable,   # type: bool
                                         none_policy   # type: NoneArgPolicy
                                         ):
    """
    Depending on none_policy and of the fact that the target parameter is nonable or not, returns a corresponding
    NonePolicy

    :param is_nonable:
    :param none_policy:
    :return:
    """
    if none_policy in {NonePolicy.VALIDATE, NonePolicy.SKIP, NonePolicy.FAIL}:
        none_policy_to_use = none_policy

    elif none_policy is NoneArgPolicy.SKIP_IF_NONABLE_ELSE_VALIDATE:
        none_policy_to_use = NonePolicy.SKIP if is_nonable else NonePolicy.VALIDATE

    elif none_policy is NoneArgPolicy.SKIP_IF_NONABLE_ELSE_FAIL:
        none_policy_to_use = NonePolicy.SKIP if is_nonable else NonePolicy.FAIL

    else:
        raise ValueError('Invalid none policy: ' + str(none_policy))
    return none_policy_to_use