def update_variables(func):
    """
    Use this decorator on Step.action implementation.

    Your action method should always return variables, or
    both variables and output.

    This decorator will update variables with output.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, tuple):
            return self.process_register(result[0], result[1])
        else:
            return self.process_register(result)

    return wrapper