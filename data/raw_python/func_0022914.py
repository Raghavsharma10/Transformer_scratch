def _serialize_item(item):
    """Internal function: serialize native types."""
    # Recursively serialize lists, tuples, and dicts.
    if isinstance(item, (list, tuple)):
        return [_serialize_item(subitem) for subitem in item]
    elif isinstance(item, dict):
        return dict([(key, _serialize_item(value))
                     for (key, value) in iteritems(item)])

    # Serialize strings.
    elif isinstance(item, string_types):
        # Replace glSomething by something (needed for WebGL commands).
        if item.startswith('gl'):
            return re.sub(r'^gl([A-Z])', lambda m: m.group(1).lower(), item)
        else:
            return item

    # Process NumPy arrays that are not buffers (typically, uniform values).
    elif isinstance(item, np.ndarray):
        return _serialize_item(item.ravel().tolist())

    # Serialize numbers.
    else:
        try:
            return np.asscalar(item)
        except Exception:
            return item