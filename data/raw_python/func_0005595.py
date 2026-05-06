def find_in_bids(filename, pattern=None, generator=False, upwards=False,
                 wildcard=True, **kwargs):
    """Find nearest file matching some criteria.

    Parameters
    ----------
    filename : instance of Path
        search the root for this file
    pattern : str
        glob string for search criteria of the filename of interest (remember
        to include '*'). The pattern is passed directly to rglob.
    wildcard : bool
        use wildcards for unspecified fields or not (if True, add "_*_" between
        fields)
    upwards : bool
        where to keep on searching upwards
    kwargs : dict


    Returns
    -------
    Path
        filename matching the pattern
    """
    if upwards and generator:
        raise ValueError('You cannot search upwards and have a generator')

    if pattern is None:
        pattern = _generate_pattern(wildcard, kwargs)

    lg.debug(f'Searching {pattern} in {filename}')

    if upwards and filename == find_root(filename):
        raise FileNotFoundError(f'Could not find file matchting {pattern} in {filename}')

    if generator:
        return filename.rglob(pattern)

    matches = list(filename.rglob(pattern))
    if len(matches) == 1:
        return matches[0]

    elif len(matches) == 0:
        if upwards:
            return find_in_bids(filename.parent, pattern=pattern, upwards=upwards)
        else:
            raise FileNotFoundError(f'Could not find file matching {pattern} in {filename}')

    else:
        matches_str = '"\n\t"'.join(str(x) for x in matches)
        raise FileNotFoundError(f'Multiple files matching "{pattern}":\n\t"{matches_str}"')