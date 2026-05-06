def _build_dataprovider_method(method, dataset, dataprovider):
    """
    Return a fabricated method that calls the dataprovider with the given
    dataset, and marshals the return value from that into params to the
    underlying test 'method'.
    :param method:
        The underlying test method.
    :type method:
        `callable`
    :param dataset:
        Tuple or GentyArgs instance containing the args of the dataset.
    :type dataset:
        `tuple` or :class:`GentyArgs`
    :param dataprovider:
        The unbound function that's responsible for generating the actual
        params that will be passed to the test function.
    :type dataprovider:
        `callable`
    :return:
        Return an unbound function that will become a test method
    :rtype:
        `function`
    """
    if isinstance(dataset, GentyArgs):
        final_args = dataset.args
        final_kwargs = dataset.kwargs
    else:
        final_args = dataset
        final_kwargs = {}

    def test_method_wrapper(my_self):
        args = dataprovider(
            my_self,
            *final_args,
            **final_kwargs
        )

        kwargs = {}

        if isinstance(args, GentyArgs):
            kwargs = args.kwargs
            args = args.args
        elif not isinstance(args, (tuple, list)):
            args = (args, )

        return method(my_self, *args, **kwargs)

    return test_method_wrapper