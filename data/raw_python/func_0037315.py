def BytesIO(*args, **kwargs):
    """BytesIO constructor shim for the async wrapper."""
    raw = sync_io.BytesIO(*args, **kwargs)
    return AsyncBytesIOWrapper(raw)