def assert_in(obj, seq, message=None, extra=None):
    """Raises an AssertionError if obj is not in seq."""
    assert obj in seq, _assert_fail_message(message, obj, seq, "is not in", extra)