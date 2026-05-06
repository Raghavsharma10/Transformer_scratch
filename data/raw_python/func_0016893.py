def find_sinks(obj):
    """
    Returns a dictionary of sink methods found on this object,
    keyed on method name. Sink methods are identified by
    (self, context) arguments on this object. For example:

    def f(self, context):
        ...

    is a sink method, but

    def f(self, ctx):
        ...

    is not.

    """
    SINK_ARGSPEC = ['self', 'context']

    return { n: m for n, m in inspect.getmembers(obj, inspect.ismethod)
        if inspect.getargspec(m)[0] == SINK_ARGSPEC }