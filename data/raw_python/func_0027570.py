def get_none_policy_text(none_policy,   # type: int
                         verbose=False  # type: bool
                         ):
    """
    Returns a user-friendly description of a NonePolicy taking into account NoneArgPolicy

    :param none_policy:
    :param verbose:
    :return:
    """
    if none_policy is NonePolicy.SKIP:
        return "accept None without performing validation" if verbose else 'SKIP'
    elif none_policy is NonePolicy.FAIL:
        return "fail on None without performing validation" if verbose else 'FAIL'
    elif none_policy is NonePolicy.VALIDATE:
        return "validate None as any other values" if verbose else 'VALIDATE'
    elif none_policy is NoneArgPolicy.SKIP_IF_NONABLE_ELSE_FAIL:
        return "accept None without validation if the argument is optional, otherwise fail on None" if verbose \
            else 'SKIP_IF_NONABLE_ELSE_FAIL'
    elif none_policy is NoneArgPolicy.SKIP_IF_NONABLE_ELSE_VALIDATE:
        return "accept None without validation if the argument is optional, otherwise validate None as any other " \
               "values" if verbose else 'SKIP_IF_NONABLE_ELSE_VALIDATE'
    else:
        raise ValueError('Invalid none_policy ' + str(none_policy))