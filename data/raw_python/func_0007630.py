def pack_list(from_, pack_type):
    """ Return the wire packed version of `from_`. `pack_type` should be some
    subclass of `xcffib.Struct`, or a string that can be passed to
    `struct.pack`. You must pass `size` if `pack_type` is a struct.pack string.
    """
    # We need from_ to not be empty
    if len(from_) == 0:
        return bytes()

    if pack_type == 'c':
        if isinstance(from_, bytes):
            # Catch Python 3 bytes and Python 2 strings
            # PY3 is "helpful" in that when you do tuple(b'foo') you get
            # (102, 111, 111) instead of something more reasonable like
            # (b'f', b'o', b'o'), so we rebuild from_ as a tuple of bytes
            from_ = [six.int2byte(b) for b in six.iterbytes(from_)]
        elif isinstance(from_, six.string_types):
            # Catch Python 3 strings and Python 2 unicode strings, both of
            # which we encode to bytes as utf-8
            # Here we create the tuple of bytes from the encoded string
            from_ = [six.int2byte(b) for b in bytearray(from_, 'utf-8')]
        elif isinstance(from_[0], six.integer_types):
            # Pack from_ as char array, where from_ may be an array of ints
            # possibly greater than 256
            def to_bytes(v):
                for _ in range(4):
                    v, r = divmod(v, 256)
                    yield r
            from_ = [six.int2byte(b) for i in from_ for b in to_bytes(i)]

    if isinstance(pack_type, six.string_types):
        return struct.pack("=" + pack_type * len(from_), *from_)
    else:
        buf = six.BytesIO()
        for item in from_:
            # If we can't pack it, you'd better have packed it yourself. But
            # let's not confuse things which aren't our Probobjs for packable
            # things.
            if isinstance(item, Protobj) and hasattr(item, "pack"):
                buf.write(item.pack())
            else:
                buf.write(item)
        return buf.getvalue()