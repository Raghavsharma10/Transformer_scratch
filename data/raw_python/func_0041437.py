def check_args(func):
    """A decorator that performs type checking using type hints at runtime::
            @check_args
            def fun(a: int):
                print(f'fun is being called with parameter {a}')
            # this will raise a TypeError describing the issue without the function being called
            fun('not an int')
    """

    @wraps(func)
    def check(*args, **kwargs):  # pylint: disable=C0111
        sig = inspect.signature(func)
        found_errors = []
        binding = None
        try:
            binding = sig.bind(*args, **kwargs)
        except TypeError as te:
            for name, metadata in sig.parameters.items():
                # Comparison with the message error as a string :(
                # Know a nicer way? Please drop me a message
                if metadata.default == inspect.Parameter.empty:
                    # copy from inspect module, it is the very same error message
                    error_in_case = 'missing a required argument: {arg!r}'.format(arg=name)
                    if str(te) == error_in_case:
                        found_errors.append(IssueDescription(
                            name, sig.parameters[name].annotation, None, True))
            # NOTE currently only find one, at most, detecting what else
            # is missing is tricky if not impossible
            if not found_errors:
                raise DetailedTypeError([IssueDescription(None, None, None, None, str(te))])
            raise DetailedTypeError(found_errors)

        for name, value in binding.arguments.items():
            if not check_type(value, sig.parameters[name].annotation):
                found_errors.append(IssueDescription(
                    name, sig.parameters[name].annotation, value, False))

        if found_errors:
            raise DetailedTypeError(found_errors)
        return func(*args, **kwargs)

    return check