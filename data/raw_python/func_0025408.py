def get_metadata_value(metadata_source, key: str) -> typing.Any:
    """Get the metadata value for the given key.

    There are a set of predefined keys that, when used, will be type checked and be interoperable with other
    applications. Please consult reference documentation for valid keys.

    If using a custom key, we recommend structuring your keys in the '<group>.<attribute>' format followed
    by the predefined keys. e.g. 'session.instrument' or 'camera.binning'.

    Also note that some predefined keys map to the metadata ``dict`` but others do not. For this reason, prefer
    using the ``metadata_value`` methods over directly accessing ``metadata``.
    """
    desc = session_key_map.get(key)
    if desc is not None:
        v = getattr(metadata_source, "session_metadata", dict())
        for k in desc['path']:
            v =  v.get(k) if v is not None else None
        return v
    desc = key_map.get(key)
    if desc is not None:
        v = getattr(metadata_source, "metadata", dict())
        for k in desc['path']:
            v =  v.get(k) if v is not None else None
        return v
    raise KeyError()