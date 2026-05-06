def call_audit(func):
    """Print a detailed audit of all calls to this function."""
    def audited_func(*args, **kwargs):
        import traceback
        stack = traceback.extract_stack()
        r = func(*args, **kwargs)
        func_name = func.__name__

        print("@depth %d, trace %s -> %s(*%r, **%r) => %r" % (
            len(stack),
            " -> ".join("%s:%d:%s" % x[0:3] for x in stack[-5:-2]),
            func_name,
            args,
            kwargs,
            r))
        return r

    return audited_func