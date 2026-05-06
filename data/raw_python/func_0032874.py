def mock_import(do_not_mock=None, **mock_kwargs):
    """
    Mocks import statements by ignoring ImportErrors
    and replacing the missing module with a Mock.

    :param str|unicode|list[str|unicode] do_not_mock: names of modules
        that should exists, and an ImportError could be raised for.
    :param mock_kwargs: kwargs for MagicMock object.
    :return: patch object
    """

    do_not_mock = _to_list(do_not_mock)

    def try_import(module_name, *args, **kwargs):
        try:
            return _builtins_import(module_name, *args, **kwargs)
        except:    # intentionally catch all exceptions
            if any((_match(module_name, prefix) for prefix in do_not_mock)):
                # This is a module we need to import,
                # so we raise the exception instead of mocking it
                raise
            # Mock external module so we can peacefully create our client
            return mock.MagicMock(**mock_kwargs)

    return mock.patch('six.moves.builtins.__import__', try_import)