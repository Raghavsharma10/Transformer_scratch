def _none_accepter(validation_callable  # type: Callable
                   ):
    # type: (...) -> Callable
    """
    Wraps the given validation callable to accept None values silently. When a None value is received by the wrapper,
    it is not passed to the validation_callable and instead this function will return True. When any other value is
    received the validation_callable is called as usual.

    Note: the created wrapper has the same same than the validation callable for more user-friendly error messages

    :param validation_callable:
    :return:
    """
    # option (a) use the `decorate()` helper method to preserve name and signature of the inner object
    # ==> NO, we want to support also non-function callable objects

    # option (b) simply create a wrapper manually
    def accept_none(x):
        if x is not None:
            # proceed with validation as usual
            return validation_callable(x)
        else:
            # value is None: skip validation
            return True

    # set a name so that the error messages are more user-friendly
    accept_none.__name__ = 'skip_on_none({})'.format(get_callable_name(validation_callable))

    return accept_none