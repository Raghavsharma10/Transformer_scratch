def watch_crc(params, ctxt, scope, stream, coord):
    """WatchCrc32 - Watch the total crc32 of the params.
    
    Example:
        The code below uses the ``WatchCrc32`` update function to update
        the ``crc`` field to the crc of the ``length`` and ``data`` fields ::

            char length;
            char data[length];
            int crc<watch=length;data, update=WatchCrc32>;
    """
    if len(params) <= 1:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "at least two arguments")
    
    to_update = params[0]

    total_data = utils.binary("")
    for param in params[1:]:
        total_data += param._pfp__build()
    
    to_update._pfp__set_value(binascii.crc32(total_data))