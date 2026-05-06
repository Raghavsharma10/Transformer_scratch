def _add_none_handler(validation_callable,  # type: Callable
                      none_policy           # type: int
                      ):
    # type: (...) -> Callable
    """
    Adds a wrapper or nothing around the provided validation_callable, depending on the selected policy

    :param validation_callable:
    :param none_policy: an int representing the None policy, see NonePolicy
    :return:
    """
    if none_policy is NonePolicy.SKIP:
        return _none_accepter(validation_callable)  # accept all None values

    elif none_policy is NonePolicy.FAIL:
        return _none_rejecter(validation_callable)  # reject all None values

    elif none_policy is NonePolicy.VALIDATE:
        return validation_callable                  # do not handle None specifically, do not wrap

    else:
        raise ValueError('Invalid none_policy : ' + str(none_policy))