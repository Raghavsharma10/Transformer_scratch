def timeCall(*funcAndArgs, **kwargs):
    r"""Return the time (in ms) it takes to call a function (the first
    argument) with the remaining arguments and `kwargs`.

    Examples:

    To find out how long ``func('foo', spam=1)`` takes to execute, do:

    ``timeCall(func, foo, spam=1)``
    """

    func, args = funcAndArgs[0], funcAndArgs[1:]
    start = time.time()
    func(*args, **kwargs)
    return time.time() - start