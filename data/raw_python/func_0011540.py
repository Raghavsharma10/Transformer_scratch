def pack_gzip(params, ctxt, scope, stream, coord):
    """``PackGZip`` - Concats the build output of all params and gzips the
    resulting data, returning a char array.

    Example: ::

        char data[0x100]<pack=PackGZip, ...>;
    """
    if len(params) == 0:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "at least one argument")
    
    built = utils.binary("")
    for param in params:
        if isinstance(param, pfp.fields.Field):
            built += param._pfp__build()
        else:
            built += param
    
    return zlib.compress(built)