def grep_iter(target, pattern, **kwargs):
    """
    Main grep function, as a memory efficient iterator.
    Note: this function does not support the 'quiet' or 'count' flags.
    :param target: Target to apply grep on. Can be a single string, an iterable, a function, or an opened file handler.
    :param pattern: Grep pattern to search.
    :param kwargs: See grep() help for more info.
    :return: Next match.
    """
    # unify flags (convert shortcuts to full name)
    __fix_args(kwargs)

    # parse the params that are relevant to this function
    f_offset = kwargs.get('byte_offset')
    f_line_number = kwargs.get('line_number')
    f_trim = kwargs.get('trim')
    f_after_context = kwargs.get('after_context')
    f_before_context = kwargs.get('before_context')
    f_only_matching = kwargs.get('only_matching')

    # if target is a callable function, call it first to get value
    if callable(target):
        target = target()

    # if we got a single string convert it to a list
    if isinstance(target, _basestring):
        target = [target]

    # calculate if need to trim end of lines
    need_to_trim_eol = not kwargs.get('keep_eol') and hasattr(target, 'readline')

    # list of previous lines, used only when f_before_context is set
    prev_lines = []

    # iterate target and grep
    for line_index, line in enumerate(target):

        # fix current line
        line = __process_line(line, need_to_trim_eol, f_trim)

        # do grap
        match, offset, endpos = __do_grep(line, pattern, **kwargs)

        # nullify return value
        value = None

        # if matched
        if match:

            # the textual part we return in response
            ret_str = line

            # if only return matching
            if f_only_matching:
                ret_str = ret_str[offset:endpos]

            # if 'before_context' is set
            if f_before_context:

                # make ret_str be a list with previous lines
                ret_str = prev_lines + [ret_str]

            # if need to return X lines after trailing context
            if f_after_context:

                # convert return string to list (unless f_before_context is set, in which case its already a list)
                if not f_before_context:
                    ret_str = [ret_str]

                # iterate X lines to read after
                for i in range(f_after_context):

                    # if target got next or readline, use next()
                    # note: unfortunately due to python files next() implementation we can't use tell and seek to
                    # restore position and not skip next matches.
                    if hasattr(target, '__next__') or hasattr(target, 'readline'):
                        try:
                            val = next(target)
                        except StopIteration:
                            break

                    # if not, try to access next item based on index (for lists)
                    else:
                        try:
                            val = target[line_index+i+1]
                        except IndexError:
                            break

                    # add value to return string
                    ret_str.append(__process_line(val, need_to_trim_eol, f_trim))

            # if requested offset, add offset + line to return list
            if f_offset:
                value = (offset, ret_str)

            # if requested line number, add offset + line to return list
            elif f_line_number:
                value = (line_index, ret_str)

            # default: add line to return list
            else:
                value = ret_str

        # maintain a list of previous lines, if the before-context option is provided
        if f_before_context:
            prev_lines.append(line)
            if len(prev_lines) > f_before_context:
                prev_lines.pop(0)

        # if we had a match return current value
        if value is not None:
            yield value

    # done iteration
    raise StopIteration