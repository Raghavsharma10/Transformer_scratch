def assert_is_instance(value, types, message=None, extra=None):
    """Raises an AssertionError if value is not an instance of type(s)."""
    assert isinstance(value, types), _assert_fail_message(
        message, value, types, "is not an instance of", extra
    )