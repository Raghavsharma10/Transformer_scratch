def posix_commonpath(paths):
    """Given a sequence of POSIX path names,
       return the longest common sub-path."""

    if not paths:
        raise ValueError('commonpath() arg is an empty sequence')

    check_arg_types('commonpath', *paths)

    if isinstance(paths[0], bytes):
        sep = b'/'
        curdir = b'.'
    else:
        sep = '/'
        curdir = '.'

    split_paths = [path.split(sep) for path in paths]

    try:
        isabs, = set(p[:1] == sep for p in paths)
    except ValueError:
        raise ValueError("Can't mix absolute and relative paths")

    split_paths = [[c for c in s if c and c != curdir] for s in split_paths]
    s_min = min(split_paths)
    s_max = max(split_paths)
    common = s_min
    for i, run_c in enumerate(s_min):
        if run_c != s_max[i]:
            common = s_min[:i]
            break

    prefix = sep if isabs else sep[:0]
    return prefix + sep.join(common)