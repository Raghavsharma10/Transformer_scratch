def get_filter_item(name: str, operation: bytes, value: bytes) -> bytes:
    """
    A field could be found for this term, try to get filter string for it.
    """
    assert isinstance(name, str)
    assert isinstance(value, bytes)
    if operation is None:
        return filter_format(b"(%s=%s)", [name, value])
    elif operation == "contains":
        assert value != ""
        return filter_format(b"(%s=*%s*)", [name, value])
    else:
        raise ValueError("Unknown search operation %s" % operation)