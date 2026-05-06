def nt_commonpath(paths):  # pylint: disable=too-many-locals
    """Given a sequence of NT path names,
       return the longest common sub-path."""

    from ntpath import splitdrive

    if not paths:
        raise ValueError('commonpath() arg is an empty sequence')

    check_arg_types('commonpath', *paths)

    if isinstance(paths[0], bytes):
        sep = b'\\'
        altsep = b'/'
        curdir = b'.'
    else:
        sep = '\\'
        altsep = '/'
        curdir = '.'

    drivesplits = [splitdrive(p.replace(altsep, sep).lower()) for p in paths]
    split_paths = [p.split(sep) for d, p in drivesplits]

    try:
        isabs, = set(p[:1] == sep for d, p in drivesplits)
    except ValueError:
        raise ValueError("Can't mix absolute and relative paths")

    # Check that all drive letters or UNC paths match. The check is made
    # only now otherwise type errors for mixing strings and bytes would not
    # be caught.
    if len(set(d for d, p in drivesplits)) != 1:
        raise ValueError("Paths don't have the same drive")

    drive, path = splitdrive(paths[0].replace(altsep, sep))
    common = path.split(sep)
    common = [c for c in common if c and c != curdir]

    split_paths = [[c for c in s if c and c != curdir] for s in split_paths]
    s_min = min(split_paths)
    s_max = max(split_paths)
    for i, run_c in enumerate(s_min):
        if run_c != s_max[i]:
            common = common[:i]
            break
    else:
        common = common[:len(s_min)]

    prefix = drive + sep if isabs else drive
    return prefix + sep.join(common)