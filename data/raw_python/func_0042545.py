def bind(context, block=False):
    """
    Given the context, returns a decorator wrapper;
    the binder replaces the wrapped func with the
    value from the context OR puts this function in
    the context with the name.
    """

    if block:
        def decorate(func):
            name = func.__name__.replace('__TK__block__', '')
            if name not in context:
                context[name] = func
            return context[name]

        return decorate

    def decorate(func):
        name = func.__name__
        if name not in context:
            context[name] = func
        return context[name]

    return decorate