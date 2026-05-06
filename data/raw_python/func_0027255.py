def assert_eq(expected, actual, message=None, tolerance=None, extra=None):
    """Raises an AssertionError if expected != actual.

    If tolerance is specified, raises an AssertionError if either
    - expected or actual isn't a number, or
    - the difference between expected and actual is larger than the tolerance.

    """
    if tolerance is None:
        assert expected == actual, _assert_fail_message(
            message, expected, actual, "!=", extra
        )
    else:
        assert isinstance(tolerance, _number_types), (
            "tolerance parameter to assert_eq must be a number: %r" % tolerance
        )
        assert isinstance(expected, _number_types) and isinstance(
            actual, _number_types
        ), (
            "parameters must be numbers when tolerance is specified: %r, %r"
            % (expected, actual)
        )

        diff = abs(expected - actual)
        assert diff <= tolerance, _assert_fail_message(
            message, expected, actual, "is more than %r away from" % tolerance, extra
        )