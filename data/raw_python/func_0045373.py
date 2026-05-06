def debug(trace=True, backtrace=1, other=None, output=sys.stderr):
    """A decorator for debugging purposes. Shows the call arguments, result
    and instructions as they are runned."""
    from pprint import pprint
    import traceback
    def debugdeco(func):
        def tracer(frame, event, arg):
            print>>output, '--- Tracer: %s, %r' % (event, arg)
            traceback.print_stack(frame, 1, output)
            return tracer
        def wrapped(*a, **k):
            print>>output
            print>>output, '--- Calling %s.%s with:' % (
                getattr(func, '__module__', ''),
                func.__name__
            )
            for i in enumerate(a):
                print>>output, '    | %s: %s' % i
            print>>output, '    | ',
            pprint(k, output)
            print>>output, '    From:'
            for i in traceback.format_stack(sys._getframe(1), backtrace):
                print>>output, i,
            if other:
                print>>output, '---      [ %r ]' % (other(func,a,k))
            if trace: sys.settrace(tracer)
            ret = func(*a, **k)
            if trace: sys.settrace(None)
            #~ a = list(a)
            print>>output, '--- %s.%s returned: %r' % (
                getattr(func, '__module__', ''),
                func.__name__,
                #~ ret not in a and ret or "ARG%s"%a.index(ret)
                ret
            )
            return ret
        return wrapped
    return debugdeco