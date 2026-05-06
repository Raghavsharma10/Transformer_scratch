def has_metadata_value(metadata_source, key: str) -> bool:
    """Return whether the metadata value for the given key exists.

    There are a set of predefined keys that, when used, will be type checked and be interoperable with other
    applications. Please consult reference documentation for valid keys.

    If using a custom key, we recommend structuring your keys in the '<group>.<attribute>' format followed
    by the predefined keys. e.g. 'session.instrument' or 'camera.binning'.

    Also note that some predefined keys map to the metadata ``dict`` but others do not. For this reason, prefer
    using the ``metadata_value`` methods over directly accessing ``metadata``.
    """
    desc = session_key_map.get(key)
    if desc is not None:
        d = getattr(metadata_source, "session_metadata", dict())
        for k in desc['path'][:-1]:
            d =  d.setdefault(k, dict()) if d is not None else None
        if d is not None:
            return desc['path'][-1] in d
    desc = key_map.get(key)
    if desc is not None:
        d = getattr(metadata_source, "metadata", dict())
        for k in desc['path'][:-1]:
            d =  d.setdefault(k, dict()) if d is not None else None
        if d is not None:
            return desc['path'][-1] in d
    raise False