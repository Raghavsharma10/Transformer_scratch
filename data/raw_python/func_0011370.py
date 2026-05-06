def FindAll(params, ctxt, scope, stream, coord, interp):
    """
    This function converts the argument data into a set of hex bytes
    and then searches the current file for all occurrences of those
    bytes. data may be any of the basic types or an array of one of
    the types. If data is an array of signed bytes, it is assumed to
    be a null-terminated string. To search for an array of hex bytes,
    create an unsigned char array and fill it with the target value. If
    the type being search for is a string, the matchcase and wholeworld
    arguments can be used to control the search (see Using Find for more
    information). method controls which search method is used from the
    following options:

    FINDMETHOD_NORMAL=0 - a normal search
    FINDMETHOD_WILDCARDS=1 - when searching for strings use wildcards '*' or '?'
    FINDMETHOD_REGEX=2 - when searching for strings use Regular Expressions

    wildcardMatchLength indicates the maximum number of characters a '*' can match when searching using wildcards. If the target is a float or double, the tolerance argument indicates that values that are only off by the tolerance value still match. If dir is 1 the find direction is down and if dir is 0 the find direction is up. start and size can be used to limit the area of the file that is searched. start is the starting byte address in the file where the search will begin and size is the number of bytes after start that will be searched. If size is zero, the file will be searched from start to the end of the file.

    The return value is a TFindResults structure. This structure contains a count variable indicating the number of matches, and a start array holding an array of starting positions, plus a size array which holds an array of target lengths. For example, use the following code to find all occurrences of the ASCII string "Test" in a file:
    """
    matches_iter = _find_helper(params, ctxt, scope, stream, coord, interp)
    matches = list(matches_iter)

    types = interp.get_types()
    res = types.TFindResults()

    res.count = len(matches)

    # python3 map doesn't return a list
    starts = list(map(lambda m: m.start()+FIND_MATCHES_START_OFFSET, matches))

    res.start = starts

    # python3 map doesn't return a list
    sizes = list(map(lambda m: m.end()-m.start(), matches))
    res.size = sizes

    return res