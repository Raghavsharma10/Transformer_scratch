def wrap(item, args=None, krgs=None, **kwargs):
    """Wraps the given item content between horizontal lines. Item can be a
    string or a function.

    **Examples**:
    ::
        qprompt.wrap("Hi, this will be wrapped.")  # String item.
        qprompt.wrap(myfunc, [arg1, arg2], {'krgk': krgv})  # Func item.
    """
    with Wrap(**kwargs):
        if callable(item):
            args = args or []
            krgs = krgs or {}
            item(*args, **krgs)
        else:
            echo(item)