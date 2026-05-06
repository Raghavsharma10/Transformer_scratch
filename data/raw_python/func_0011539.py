def packer_gzip(params, ctxt, scope, stream, coord):
    """``PackerGZip`` - implements both unpacking and packing. Can be used
    as the ``packer`` for a field. When packing, concats the build output
    of all params and gzip-compresses the result. When unpacking, concats
    the build output of all params and gzip-decompresses the result.
    
    Example:

        The code below specifies that the ``data`` field is gzipped
        and that once decompressed, should be parsed with ``PACK_TYPE``.
        When building the ``PACK_TYPE`` structure, ``data`` will be updated
        with the compressed data.::

            char data[0x100]<packer=PackerGZip, packtype=PACK_TYPE>;

    :pack: True if the data should be packed, false if it should be unpacked
    :data: The data to operate on
    :returns: An array
    """
    if len(params) <= 1:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "at least two arguments")

    # to gzip it (pack it)
    if params[0]:
        return pack_gzip(params[1:], ctxt, scope, stream, coord)
    else:
        return unpack_gzip(params[1:], ctxt, scope, stream, coord)