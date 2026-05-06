def read(parser, stream):
    """
    Return an AST from the input ES5 stream.

    Arguments

    parser
        A parser instance.
    stream
        Either a stream object or a callable that produces one.  The
        stream object to read from; its 'read' method will be invoked.

        If a callable was provided, the 'close' method on its return
        value will be called to close the stream.
    """

    source = stream() if callable(stream) else stream
    try:
        text = source.read()
        stream_name = getattr(source, 'name', None)
        try:
            result = parser(text)
        except ECMASyntaxError as e:
            error_name = repr_compat(stream_name or source)
            raise type(e)('%s in %s' % (str(e), error_name))
    finally:
        if callable(stream):
            source.close()

    result.sourcepath = stream_name
    return result