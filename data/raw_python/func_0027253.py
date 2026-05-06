def assert_is_not(expected, actual, message=None, extra=None):
    """Raises an AssertionError if expected is actual."""
    assert expected is not actual, _assert_fail_message(
        message, expected, actual, "is", extra
    )