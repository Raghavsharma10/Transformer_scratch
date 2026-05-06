def unpack_flags(value, flags):
    """Multiple flags might be packed in the same field."""
    try:
        return [flags[value]]
    except KeyError:
        return [flags[k] for k in sorted(flags.keys()) if k & value > 0]