def RawParserUnparserFactory(parser_name, parse_callable, *unparse_callables):
    """
    Produces a callable object that also has callable attributes that
    passes its first argument to the parent callable.
    """

    def build_unparse(f):
        @wraps(f)
        def unparse(self, source, *a, **kw):
            node = parse_callable(source)
            return f(node, *a, **kw)
        # a dumb and lazy docstring replacement
        unparse.__doc__ = f.__doc__.replace(
            'ast\n        The AST ',
            'source\n        The source ',
        )
        return unparse

    def build_parse(f):
        @wraps(f)
        def parse(self, source):
            return f(source)
        parse.__name__ = parser_name
        parse.__qualname__ = parser_name
        return parse

    callables = {f.__name__: build_unparse(f) for f in unparse_callables}
    callables['__call__'] = build_parse(parse_callable)
    callables['__module__'] = PKGNAME
    return type(parser_name, (object,), callables)()