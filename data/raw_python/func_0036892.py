def _rdumpq(q,size,value,encoding=None):
    """Dump value as a tnetstring, to a deque instance, last chunks first.

    This function generates the tnetstring representation of the given value,
    pushing chunks of the output onto the given deque instance.  It pushes
    the last chunk first, then recursively generates more chunks.

    When passed in the current size of the string in the queue, it will return
    the new size of the string in the queue.

    Operating last-chunk-first makes it easy to calculate the size written
    for recursive structures without having to build their representation as
    a string.  This is measurably faster than generating the intermediate
    strings, especially on deeply nested structures.
    """
    write = q.appendleft
    if value is None:
        write("0:~")
        return size + 3
    if value is True:
        write("4:true!")
        return size + 7
    if value is False:
        write("5:false!")
        return size + 8
    if isinstance(value,(int,long)):
        data = str(value) 
        ldata = len(data)
        span = str(ldata)
        write("#")
        write(data)
        write(":")
        write(span)
        return size + 2 + len(span) + ldata
    if isinstance(value,(float,)):
        #  Use repr() for float rather than str().
        #  It round-trips more accurately.
        #  Probably unnecessary in later python versions that
        #  use David Gay's ftoa routines.
        data = repr(value) 
        ldata = len(data)
        span = str(ldata)
        write("^")
        write(data)
        write(":")
        write(span)
        return size + 2 + len(span) + ldata
    if isinstance(value,str):
        lvalue = len(value)
        span = str(lvalue)
        write(",")
        write(value)
        write(":")
        write(span)
        return size + 2 + len(span) + lvalue
    if isinstance(value,(list,tuple,)):
        write("]")
        init_size = size = size + 1
        for item in reversed(value):
            size = _rdumpq(q,size,item,encoding)
        span = str(size - init_size)
        write(":")
        write(span)
        return size + 1 + len(span)
    if isinstance(value,dict):
        write("}")
        init_size = size = size + 1
        for (k,v) in value.iteritems():
            size = _rdumpq(q,size,v,encoding)
            size = _rdumpq(q,size,k,encoding)
        span = str(size - init_size)
        write(":")
        write(span)
        return size + 1 + len(span)
    if isinstance(value,unicode):
        if encoding is None:
            raise ValueError("must specify encoding to dump unicode strings")
        value = value.encode(encoding)
        lvalue = len(value)
        span = str(lvalue)
        write(",")
        write(value)
        write(":")
        write(span)
        return size + 2 + len(span) + lvalue
    raise ValueError("unserializable object")