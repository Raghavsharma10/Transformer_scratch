def merge_two_dictionaries(a, b, merge_lists=False):
    # type: (DictUpperBound, DictUpperBound, bool) -> DictUpperBound
    """Merges b into a and returns merged result

    NOTE: tuples and arbitrary objects are not handled as it is totally ambiguous what should happen

    Args:
        a (DictUpperBound): dictionary to merge into
        b (DictUpperBound): dictionary to merge from
        merge_lists (bool): Whether to merge lists (True) or replace lists (False). Default is False.

    Returns:
        DictUpperBound: Merged dictionary
    """
    key = None
    # ## debug output
    # sys.stderr.write('DEBUG: %s to %s\n' %(b,a))
    try:
        if a is None or isinstance(a, (six.string_types, six.text_type, six.integer_types, float)):
            # border case for first run or if a is a primitive
            a = b
        elif isinstance(a, list):
            # lists can be appended or replaced
            if isinstance(b, list):
                if merge_lists:
                    # merge lists
                    a.extend(b)
                else:
                    # replace list
                    a = b
            else:
                # append to list
                a.append(b)
        elif isinstance(a, (dict, UserDict)):
            # dicts must be merged
            if isinstance(b, (dict, UserDict)):
                for key in b:
                    if key in a:
                        a[key] = merge_two_dictionaries(a[key], b[key], merge_lists=merge_lists)
                    else:
                        a[key] = b[key]
            else:
                raise ValueError('Cannot merge non-dict "%s" into dict "%s"' % (b, a))
        else:
            raise ValueError('NOT IMPLEMENTED "%s" into "%s"' % (b, a))
    except TypeError as e:
        raise ValueError('TypeError "%s" in key "%s" when merging "%s" into "%s"' % (e, key, b, a))
    return a