def assert_ge(left, right, message=None, extra=None):
    """Raises an AssertionError if left_hand < right_hand."""
    assert left >= right, _assert_fail_message(message, left, right, "<", extra)