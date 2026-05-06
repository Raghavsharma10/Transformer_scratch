def _merge(d, u):
    """Merge two dictionaries (or DotDicts) together.

    Args:
          d: The dictionary/DotDict to merge into.
          u: The source of the data to merge.
    """
    for k, v in u.items():
        # if we have a mapping, recursively merge the values
        if isinstance(v, collections.Mapping):
            d[k] = _merge(d.get(k, {}), v)

        # if d (the dict to merge into) is a dict, just add the
        # value to the dict.
        elif isinstance(d, collections.MutableMapping):
            d[k] = v

        # otherwise if d (the dict to merge into) is not a dict (e.g. when
        # recursing into it, `d.get(k, {})` may not be a dict), then do what
        # `update` does and prefer the new value.
        #
        # this means that something like `{'foo': 1}` when updated with
        # `{'foo': {'bar': 1}}` would have the original value (`1`) overwritten
        # and would become: `{'foo': {'bar': 1}}`
        else:
            d = {k: v}

    return d