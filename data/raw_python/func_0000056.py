def spy2(fn):  # type: (...) -> None
    """Spy usage of given `fn`.

    Patches the module, class or object `fn` lives in, so that all
    interactions can be recorded; otherwise executes `fn` as before, so
    that all side effects happen as before.

    E.g.::

        import time
        spy(time.time)
        do_work(...)  # nothing injected, uses global patched `time` module
        verify(time).time()

    Note that builtins often cannot be patched because they're read-only.


    """
    if isinstance(fn, str):
        answer = get_obj(fn)
    else:
        answer = fn

    when2(fn, Ellipsis).thenAnswer(answer)