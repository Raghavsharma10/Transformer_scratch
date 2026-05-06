def genty_dataset(*args, **kwargs):
    """Decorator defining data sets to provide to a test.

    Inspired by http://sebastian-bergmann.de/archives/
    702-Data-Providers-in-PHPUnit-3.2.html

    The canonical way to call @genty_dataset, with each argument each
    representing a data set to be injected in the test method call:
        @genty_dataset(
            ('a1', 'b1'),
            ('a2', 'b2'),
        )
        def test_some_function(a, b)
            ...

    If the test function takes only one parameter, you can replace the tuples
    by a single value. So instead of the more verbose:
        @genty_dataset(
            ('c1',),
            ('c2',),
        )
        def test_some_other_function(c)
            ...

    One can write:
        @genty_dataset('c1', 'c2')
        def test_some_other_function(c)
            ...

    For each set of arguments, a suffix identifying that argument set is
    built by concatenating the string representation of the arguments
    together. You can control the test names for each data set by passing
    the data sets as keyword args, where the keyword is the desired suffix.
    For example:
        @genty_dataset(
            ('a1', 'b1),
        )
        def test_function(a, b)
            ...
    produces a test named 'test_function_for_a1_and_b1', while
        @genty_dataset(
            happy_path=('a1', 'b1'),
        )
        def test_function(a, b)
            ...
    produces a test named test_function_for_happy_path. These are just
    parameters to a method call, so one can have unnamed args first
    followed by keyword args
        @genty_dataset(
            ('x', 'y'),
            ('p', 'q'),
            Monday=('a1', 'b1'),
            Tuesday=('t1', 't2'),
        )
        def test_function(a, b)
            ...

    Finally, datasets can be chained. Useful for example if there are
    distinct sets of params that make sense (cleaner, more readable, or
    semantically nicer) if kept separate. A fabricated example:

        @genty_dataset(
            *([i for i in range(10)] + [(i, i) for i in range(10)])
        )
        def test_some_other_function(param1, param2=None)
            ...

        -- vs --

        @genty_dataset(*[i for i in range(10)])
        @genty_dataset(*[(i, i) for i in range(10)])
        def test_some_other_function(param1, param2=None)
            ...

    If the names of datasets conflict across chained genty_datasets, the
    key&value pair from the outer (first) decorator will override the
    data from the inner.

    :param args:
        Tuple of unnamed data sets.
    :type args:
        `tuple` of varies
    :param kwargs:
        Dict of pre-named data sets.
    :type kwargs:
        `dict` of `unicode` to varies
    """
    datasets = _build_datasets(*args, **kwargs)

    def wrap(test_method):
        # Save the datasets in the test method. This data will be consumed
        # by the @genty decorator.
        if not hasattr(test_method, 'genty_datasets'):
            test_method.genty_datasets = OrderedDict()

        test_method.genty_datasets.update(datasets)

        return test_method
    return wrap