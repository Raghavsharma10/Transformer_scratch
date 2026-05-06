def _contextualize(contextFactory, contextReceiver):
    """
    Invoke a callable with an argument derived from the current execution
    context (L{twisted.python.context}), or automatically created if none is
    yet present in the current context.

    This function, with a better name and documentation, should probably be
    somewhere in L{twisted.python.context}.  Calling context.get() and
    context.call() individually is perilous because you always have to handle
    the case where the value you're looking for isn't present; this idiom
    forces you to supply some behavior for that case.

    @param contextFactory: An object which is both a 0-arg callable and
    hashable; used to look up the value in the context, set the value in the
    context, and create the value (by being called).

    @param contextReceiver: A function that receives the value created or
    identified by contextFactory.  It is a 1-arg callable object, called with
    the result of calling the contextFactory, or retrieving the contextFactory
    from the context.
    """
    value = context.get(contextFactory, _NOT_SPECIFIED)
    if value is not _NOT_SPECIFIED:
        return contextReceiver(value)
    else:
        return context.call({contextFactory: contextFactory()},
                            _contextualize, contextFactory, contextReceiver)