def FlowManager(start_string, end_string):
    """
    A factory for creating context managers for standard code constructions,
    such as if statements and functions.

    :param str start_string: A format string for the beginning of this code
    structure
    :param str end_string: A format string for the end of this code structure

    The constructor of the returned class takes as arguments a `CodeWriter`
    instance followed by keyword arguments that will be used to format
    `start_string` and `end_string`
    """
    class Inner(object):
        def __init__(self, f, **kwargs):
            self.f = f
            self.kwargs = kwargs

        def __enter__(self):
            self.f.writeln(start_string.format(**self.kwargs))
            self.f.indent()

        def __exit__(self, type, value, traceback):
            self.f.deindent()
            self.f.writeln(end_string.format(**self.kwargs))
    return Inner