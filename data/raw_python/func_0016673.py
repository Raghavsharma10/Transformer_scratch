def uniform_iterator(sequence):
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, abc.Mapping):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)