def assert_is_not_substring(substring, subject, message=None, extra=None):
    """Raises an AssertionError if substring is a substring of subject."""
    assert (
        (subject is not None)
        and (substring is not None)
        and (subject.find(substring) == -1)
    ), _assert_fail_message(message, substring, subject, "is in", extra)