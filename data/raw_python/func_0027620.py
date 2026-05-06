def _failure_raiser(validation_callable,   # type: Callable
                    failure_type=None,     # type: Type[WrappingFailure]
                    help_msg=None,         # type: str
                    **kw_context_args):
    # type: (...) -> Callable
    """
    Wraps the provided validation function so that in case of failure it raises the given failure_type or a WrappingFailure
    with the given help message.

    :param validation_callable:
    :param failure_type: an optional subclass of `WrappingFailure` that should be raised in case of failure, instead of
    `WrappingFailure`.
    :param help_msg: an optional string help message for the raised `WrappingFailure` (if no failure_type is provided)
    :param kw_context_args: optional context arguments for the custom failure message
    :return:
    """

    # check failure type
    if failure_type is not None and help_msg is not None:
        raise ValueError('Only one of failure_type and help_msg can be set at the same time')

    # convert mini-lambdas to functions if needed
    validation_callable = as_function(validation_callable)

    # create wrapper
    # option (a) use the `decorate()` helper method to preserve name and signature of the inner object
    # ==> NO, we want to support also non-function callable objects

    # option (b) simply create a wrapper manually
    def raiser(x):
        """ Wraps validation_callable to raise a failure_type_or_help_msg in case of failure """

        try:
            # perform validation
            res = validation_callable(x)

        except Exception as e:
            # no need to raise from e since the __cause__ is already set in the constructor: we can safely commonalize
            res = e

        if not result_is_success(res):
            typ = failure_type or WrappingFailure
            exc = typ(wrapped_func=validation_callable, wrong_value=x, validation_outcome=res,
                      help_msg=help_msg, **kw_context_args)
            raise exc

    # set a name so that the error messages are more user-friendly

    # NO, Do not include the callable type or error message in the name since it is only used in error messages where
    # they will appear anyway !
    # ---
    # if help_msg or failure_type:
    #     raiser.__name__ = 'failure_raiser({}, {})'.format(get_callable_name(validation_callable),
    #                                                       help_msg or failure_type.__name__)
    # else:
    # ---
    # raiser.__name__ = 'failure_raiser({})'.format(get_callable_name(validation_callable))
    raiser.__name__ = get_callable_name(validation_callable)
    # Note: obviously this can hold as long as we do not check the name of this object in any other context than
    # raising errors. If we want to support this, then creating a callable object with everything in the fields will be
    # probably more appropriate so that error messages will be able to display the inner name, while repr() will still
    # say that this is a failure raiser.
    # TODO consider transforming failure_raiser into a class (see comment above)

    return raiser