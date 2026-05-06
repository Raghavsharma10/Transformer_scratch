def format_args_in_redis_protocol(*args):
    """Formats arguments into redis protocol...

    This function makes and returns a string/buffer corresponding to
    given arguments formated with the redis protocol.

    integer, text, string or binary types are automatically converted
    (using utf8 if necessary).

    More informations about the protocol: http://redis.io/topics/protocol

    Args:
        *args: full redis command as variable length argument list

    Returns:
        binary string (arguments in redis protocol)

    Examples:
        >>> format_args_in_redis_protocol("HSET", "key", "field", "value")
        '*4\r\n$4\r\nHSET\r\n$3\r\nkey\r\n$5\r\nfield\r\n$5\r\nvalue\r\n'
    """
    buf = WriteBuffer()
    l = "*%d\r\n" % len(args)  # noqa: E741
    if six.PY2:
        buf.append(l)
    else:  # pragma: no cover
        buf.append(l.encode('utf-8'))
    for arg in args:
        if isinstance(arg, six.text_type):
            # it's a unicode string in Python2 or a standard (unicode)
            # string in Python3, let's encode it in utf-8 to get raw bytes
            arg = arg.encode('utf-8')
        elif isinstance(arg, six.string_types):
            # it's a basestring in Python2 => nothing to do
            pass
        elif isinstance(arg, six.binary_type):  # pragma: no cover
            # it's a raw bytes string in Python3 => nothing to do
            pass
        elif isinstance(arg, six.integer_types):
            tmp = "%d" % arg
            if six.PY2:
                arg = tmp
            else:  # pragma: no cover
                arg = tmp.encode('utf-8')
        elif isinstance(arg, WriteBuffer):
            # it's a WriteBuffer object => nothing to do
            pass
        else:
            raise Exception("don't know what to do with %s" % type(arg))
        l = "$%d\r\n" % len(arg)  # noqa: E741
        if six.PY2:
            buf.append(l)
        else:  # pragma: no cover
            buf.append(l.encode('utf-8'))
        buf.append(arg)
        buf.append(b"\r\n")
    return buf