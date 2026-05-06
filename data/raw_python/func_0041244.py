def _do_denormalize (version_tuple):
    """separate action function to allow for the memoize decorator.  Lists,
    the most common thing passed in to the 'denormalize' below are not hashable.
    """
    version_parts_list = []
    for parts_tuple in itertools.imap(None,*([iter(version_tuple)]*4)):
        version_part = ''.join(fn(x) for fn, x in
                               zip(_denormalize_fn_list, parts_tuple))
        if version_part:
            version_parts_list.append(version_part)
    return '.'.join(version_parts_list)