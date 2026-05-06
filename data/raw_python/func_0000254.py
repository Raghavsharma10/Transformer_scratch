def _transaction_func(entering, exiting, using):
    """
    Takes 3 things, an entering function (what to do to start this block of
    transaction management), an exiting function (what to do to end it, on both
    success and failure, and using which can be: None, indiciating transaction
    should occur on all defined servers, or a callable, indicating that using
    is None and to return the function already wrapped.

    Returns either a Transaction objects, which is both a decorator and a
    context manager, or a wrapped function, if using is a callable.
    """
    # Note that although the first argument is *called* `using`, it
    # may actually be a function; @autocommit and @autocommit('foo')
    # are both allowed forms.
    if callable(using):
        return Transaction(entering, exiting, None)(using)
    return Transaction(entering, exiting, using)