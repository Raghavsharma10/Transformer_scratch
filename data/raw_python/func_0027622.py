def _none_rejecter(validation_callable  # type: Callable
                   ):
    # type: (...) -> Callable
    """
    Wraps the given validation callable to reject None values. When a None value is received by the wrapper,
    it is not passed to the validation_callable and instead this function will raise a WrappingFailure. When any other value is
    received the validation_callable is called as usual.

    :param validation_callable:
    :return:
    """
    # option (a) use the `decorate()` helper method to preserve name and signature of the inner object
    # ==> NO, we want to support also non-function callable objects

    # option (b) simply create a wrapper manually
    def reject_none(x):
        if x is not None:
            return validation_callable(x)
        else:
            raise ValueIsNone(wrong_value=x)

    # set a name so that the error messages are more user-friendly ==> NO ! here we want to see the checker
    reject_none.__name__ = 'reject_none({})'.format(get_callable_name(validation_callable))

    return reject_none