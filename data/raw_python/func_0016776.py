def refresh_index(meta, index) -> None:
    """Recalculate the projection, hash_key, and range_key for the given index.

    :param meta: model.Meta to find columns by name
    :param index: The index to refresh
    """
    # All projections include model + index keys
    projection_keys = set.union(meta.keys, index.keys)

    proj = index.projection
    mode = proj["mode"]

    if mode == "keys":
        proj["included"] = projection_keys
    elif mode == "all":
        proj["included"] = meta.columns
    elif mode == "include":  # pragma: no branch
        if all(isinstance(p, str) for p in proj["included"]):
            proj["included"] = set(meta.columns_by_name[n] for n in proj["included"])
        else:
            proj["included"] = set(proj["included"])
        proj["included"].update(projection_keys)

    if proj["strict"]:
        proj["available"] = proj["included"]
    else:
        proj["available"] = meta.columns