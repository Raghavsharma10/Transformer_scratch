def get_function_host(fn):
    """Destructure a given function into its host and its name.

    The 'host' of a function is a module, for methods it is usually its
    instance or its class. This is safe only for methods, for module wide,
    globally declared names it must be considered experimental.

    For all reasonable fn: ``getattr(*get_function_host(fn)) == fn``

    Returns tuple (host, fn-name)
    Otherwise should raise TypeError
    """

    obj = None
    try:
        name = fn.__name__
        obj = fn.__self__
    except AttributeError:
        pass

    if obj is None:
        # Due to how python imports work, everything that is global on a module
        # level must be regarded as not safe here. For now, we go for the extra
        # mile, TBC, because just specifying `os.path.exists` would be 'cool'.
        #
        # TLDR;:
        # E.g. `inspect.getmodule(os.path.exists)` returns `genericpath` bc
        # that's where `exists` is defined and comes from. But from the point
        # of view of the user `exists` always comes and is used from `os.path`
        # which points e.g. to `ntpath`. We thus must patch `ntpath`.
        # But that's the same for most imports::
        #
        #     # b.py
        #     from a import foo
        #
        # Now asking `getmodule(b.foo)` it tells you `a`, but we access and use
        # `b.foo` and we therefore must patch `b`.

        obj, name = find_invoking_frame_and_try_parse()
        # safety check!
        assert getattr(obj, name) == fn


    return obj, name