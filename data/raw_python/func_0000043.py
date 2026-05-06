def verify(obj, times=1, atleast=None, atmost=None, between=None,
           inorder=False):
    """Central interface to verify interactions.

    `verify` uses a fluent interface::

        verify(<obj>, times=2).<method_name>(<args>)

    `args` can be as concrete as necessary. Often a catch-all is enough,
    especially if you're working with strict mocks, bc they throw at call
    time on unwanted, unconfigured arguments::

        from mockito import ANY, ARGS, KWARGS
        when(manager).add_tasks(1, 2, 3)
        ...
        # no need to duplicate the specification; every other argument pattern
        # would have raised anyway.
        verify(manager).add_tasks(1, 2, 3)  # duplicates `when`call
        verify(manager).add_tasks(*ARGS)
        verify(manager).add_tasks(...)       # Py3
        verify(manager).add_tasks(Ellipsis)  # Py2

    """

    if isinstance(obj, str):
        obj = get_obj(obj)

    verification_fn = _get_wanted_verification(
        times=times, atleast=atleast, atmost=atmost, between=between)
    if inorder:
        verification_fn = verification.InOrder(verification_fn)

    # FIXME?: Catch error if obj is neither a Mock nor a known stubbed obj
    theMock = _get_mock_or_raise(obj)

    class Verify(object):
        def __getattr__(self, method_name):
            return invocation.VerifiableInvocation(
                theMock, method_name, verification_fn)

    return Verify()