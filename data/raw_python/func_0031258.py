def _merge_dicts(first, second):
    """Merge the 'second' multiple-dictionary into the 'first' one."""
    new = deepcopy(first)
    for k, v in second.items():
        if isinstance(v, dict) and v:
            ret = _merge_dicts(new.get(k, dict()), v)
            new[k] = ret
        else:
            new[k] = second[k]
    return new