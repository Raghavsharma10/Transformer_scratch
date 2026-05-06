def genty_dataprovider(builder_function):
    """Decorator defining that this test gets parameters from the given
    build_function.

    :param builder_function:
        A callable that returns parameters that will be passed to the method
        decorated by this decorator.

        If the builder_function returns a tuple or list, then that will be
        passed as *args to the decorated method.

        If the builder_function returns a :class:`GentyArgs`, then that will
        be used to pass *args and **kwargs to the decorated method.

        Any other return value will be treated as a single parameter, and
        passed as such to the decorated method.
    :type builder_function:
        `callable`
    """
    datasets = getattr(builder_function, 'genty_datasets', {None: ()})

    def wrap(test_method):
        # Save the data providers in the test method. This data will be
        # consumed by the @genty decorator.
        if not hasattr(test_method, 'genty_dataproviders'):
            test_method.genty_dataproviders = []

        test_method.genty_dataproviders.append(
            (builder_function, datasets),
        )

        return test_method
    return wrap