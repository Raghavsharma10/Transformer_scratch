def StringIO(*args, **kwargs):
    """StringIO constructor shim for the async wrapper."""
    raw = sync_io.StringIO(*args, **kwargs)
    return AsyncStringIOWrapper(raw)