def keys_values(data, *keys):
    """Get an entry as a list from a dict. Provide a fallback key."""
    values = []
    if is_mapping(data):
        for key in keys:
            if key in data:
                values.extend(ensure_list(data[key]))
    return values