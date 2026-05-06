def __do_grep(curr_line, pattern, **kwargs):
    """
    Do grep on a single string.
    See 'grep' docs for info about kwargs.
    :param curr_line: a single line to test.
    :param pattern: pattern to search.
    :return: (matched, position, end_position).
    """
    # currently found position
    position = -1
    end_pos = -1

    # check if fixed strings mode
    if kwargs.get('fixed_strings'):

        # if case insensitive fix case
        if kwargs.get('ignore_case'):
            pattern = pattern.lower()
            curr_line = curr_line.lower()

        # if pattern is a single string, match it:
        pattern_len = 0
        if isinstance(pattern, _basestring):
            position = curr_line.find(pattern)
            pattern_len = len(pattern)

        # if not, treat it as a list of strings and match any
        else:
            for p in pattern:
                position = curr_line.find(p)
                pattern_len = len(p)
                if position != -1:
                    break

        # calc end position
        end_pos = position + pattern_len

        # check if need to match whole words
        if kwargs.get('words') and position != -1:

            foundpart = (' ' + curr_line + ' ')[position:position+len(pattern)+2]
            if _is_part_of_word(foundpart[0]):
                position = -1
            elif _is_part_of_word(foundpart[-1]):
                position = -1

    # if not fixed string, it means its a regex
    else:

        # set regex flags
        flags = kwargs.get('regex_flags') or 0
        flags |= re.IGNORECASE if kwargs.get('ignore_case') else 0

        # add whole-words option
        if kwargs.get('words'):
            pattern = r'\b' + pattern + r'\b'

        # do search
        result = re.search(pattern, curr_line, flags)

        # if found, set position
        if result:
            position = result.start()
            end_pos = result.end()

    # check if need to match whole line
    if kwargs.get('line') and (position != 0 or end_pos != len(curr_line)):
        position = -1

    # parse return value
    matched = position != -1

    # if invert flag is on, invert value
    if kwargs.get('invert'):
        matched = not matched

    # if position is -1 reset end pos as well
    if not matched:
        end_pos = -1

    # return result
    return matched, position, end_pos