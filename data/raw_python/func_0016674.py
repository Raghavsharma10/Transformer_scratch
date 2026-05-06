def uniform_get(sequence, index, default=None):
    """Uniform `dict`/`list` item getter, where `index` is interpreted as a key
    for maps and as numeric index for lists."""

    if isinstance(sequence, abc.Mapping):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default