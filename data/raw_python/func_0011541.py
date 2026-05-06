def watch_length(params, ctxt, scope, stream, coord):
    """WatchLength - Watch the total length of each of the params.
    
    Example:
        The code below uses the ``WatchLength`` update function to update
        the ``length`` field to the length of the ``data`` field ::

            int length<watch=data, update=WatchLength>;
            char data[length];
    """
    if len(params) <= 1:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "at least two arguments")
    
    to_update = params[0]

    total_size = 0
    for param in params[1:]:
        total_size += param._pfp__width()
    
    to_update._pfp__set_value(total_size)