def build_options(payload, options, maxsize = 576, overload = OVERLOAD_FILE | OVERLOAD_SNAME, allowpartial = True):
    '''
    Split a list of options
    
    This is the reverse operation of `reassemble_options`, it splits `dhcp_option` into
    `dhcp_option_partial` if necessary, and set overload option if field overloading is
    used.
    
    :param options: a list of `dhcp_option`
    
    :param maxsize: Limit the maximum DHCP message size. If options cannot fit into the DHCP
                    message, specified fields are overloaded for options. If options cannot
                    fit after overloading, extra options are DROPPED if allowpartial = True.
                    
                    It is important to sort the dhcp options by priority.
    
    :param overload: fields that are allowed to be overloaded
    
    :param allowpartial: When options cannot fit into the DHCP message, allow the rest options
                         to be dropped.
                         
    :return: Number of options that are dropped i.e. `options[:-return_value]` are dropped
    '''
    if maxsize < 576:
        maxsize = 576
    max_options_size = maxsize - 240
    # Ignore OPTION_PAD and OPTION_END
    options = [o for o in options if o.tag not in (OPTION_PAD, OPTION_END)]
    # Only preserve data
    option_data = [(o.tag, o._tobytes()[2:]) for o in options]
    def split_options(option_data, limits):
        """
        Split options into multiple fields
        
        :param option_data: list of (tag, data) pair
        
        :param limits: list of int for limit of each field (excluding PAD and END)
        
        :return: number of options that are dropped
        """
        # List of (dhcp_option_partial, option_not_finished)
        partial_options = []
        buffers = [0]
        if not options:
            return ([], 0)
        def create_result():
            # Remove any unfinished partial options
            while partial_options and partial_options[-1][1]:
                partial_options.pop()
            buffers.append(len(partial_options))
            r = [[po for po,_ in partial_options[buffers[i]:buffers[i+1]]] for i in range(0, len(buffers) - 1)]
            # Remove empty fields
            while r and not r[-1]:
                r.pop()
            return r
        # Current field used size
        current_size = 0
        limit_iter = iter(limits)
        try:
            next_limit = next(limit_iter)
        except (StopIteration, GeneratorExit):
            return ([], False)
        for i, (tag, data) in enumerate(option_data):
            # Current used data size
            data_size = 0
            # Do not split very small options on boundary, this may prevent some broken DHCP clients/servers
            # to cause problem
            nosplit = (len(data) <= 32)
            while True:
                # next partial option size should be:
                # 1. no more than the current field limit (minus 1-byte tag and 1-byte length)
                # 2. no more than the single dhcp_option_partial data limit (255 due to single byte length)
                # 3. no more than the rest data size
                next_size = min(next_limit - current_size - 2, 255, len(data) - data_size)
                if next_size < 0 or (next_size == 0 and data_size < len(data)) \
                        or (next_size < len(data) - data_size and nosplit):
                    # Cannot put this part of data on the current field, find the next field
                    try:
                        next_limit = next(limit_iter)
                    except (StopIteration, GeneratorExit):
                        return (create_result(), len(option_data) - i)
                    # Record field boundary
                    buffers.append(len(partial_options))
                    current_size = 0
                else:
                    # Put this partial option on current field
                    partial_options.append((dhcp_option_partial(tag = tag, data = data[data_size : data_size + next_size]),
                                            (next_size < len(data) - data_size)))
                    data_size += next_size
                    current_size += next_size + 2
                    if data_size >= len(data):
                        # finished current option
                        break
        return (create_result(), 0)
    # First try to fit all options in options field
    # preserve a byte for OPTION_END
    result, not_finished = split_options(option_data, [max_options_size - 1])
    if not_finished:
        if overload & (OVERLOAD_FILE | OVERLOAD_SNAME):
            # Try overload
            # minus a overload option (1-byte tag, 1-byte lenght, 1-byte dhcp_overload) and 1-byte OPTION_END
            limits = [max_options_size - 4]
            if overload & OVERLOAD_FILE:
                # preserve a byte for OPTION_END
                limits.append(127)
            if overload & OVERLOAD_SNAME:
                # preserve a byte for OPTION_END
                limits.append(63)
            result2, not_finished2 = split_options(option_data, limits)
            # Only overload if we have a better result
            if len(result2) > 1:
                result = result2
                not_finished = not_finished2
    if not allowpartial and not_finished:
        raise ValueError("%d options cannot fit into a DHCP message" % (not_finished,))
    if not result:
        return not_finished
    elif len(result) <= 1:
        # No overload
        payload.options = result[0] + [dhcp_option_partial(tag = OPTION_END)]
    else:
        overload_option = 0
        if len(result) >= 2 and result[1]:
            overload_option |= OVERLOAD_FILE
            # overload file field
            payload.file = dhcp_option_partial[0].tobytes(result[1] + [dhcp_option_partial(tag = OPTION_END)])
        if len(result) >= 3 and result[2]:
            overload_option |= OVERLOAD_SNAME
            # overload sname field
            payload.sname = dhcp_option_partial[0].tobytes(result[2] + [dhcp_option_partial(tag = OPTION_END)])
        # Put an overload option before any other options
        payload.options = [dhcp_option_partial(tag = OPTION_OVERLOAD, data = dhcp_overload.tobytes(overload_option))] \
                        + result[0] + [dhcp_option_partial(tag = OPTION_END)]
    return not_finished