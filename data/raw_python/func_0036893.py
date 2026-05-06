def _gdumps(value,encoding):
    """Generate fragments of value dumped as a tnetstring.

    This is the naive dumping algorithm, implemented as a generator so that
    it's easy to pass to "".join() without building a new list.

    This is mainly here for comparison purposes; the _rdumpq version is
    measurably faster as it doesn't have to build intermediate strins.
    """
    if value is None:
        yield "0:~"
    elif value is True:
        yield "4:true!"
    elif value is False:
        yield "5:false!"
    elif isinstance(value,(int,long)):
        data = str(value) 
        yield str(len(data))
        yield ":"
        yield data
        yield "#"
    elif isinstance(value,(float,)):
        data = repr(value) 
        yield str(len(data))
        yield ":"
        yield data
        yield "^"
    elif isinstance(value,(str,)):
        yield str(len(value))
        yield ":"
        yield value
        yield ","
    elif isinstance(value,(list,tuple,)):
        sub = []
        for item in value:
            sub.extend(_gdumps(item))
        sub = "".join(sub)
        yield str(len(sub))
        yield ":"
        yield sub
        yield "]"
    elif isinstance(value,(dict,)):
        sub = []
        for (k,v) in value.iteritems():
            sub.extend(_gdumps(k))
            sub.extend(_gdumps(v))
        sub = "".join(sub)
        yield str(len(sub))
        yield ":"
        yield sub
        yield "}"
    elif isinstance(value,(unicode,)):
        if encoding is None:
            raise ValueError("must specify encoding to dump unicode strings")
        value = value.encode(encoding)
        yield str(len(value))
        yield ":"
        yield value
        yield ","
    else:
        raise ValueError("unserializable object")