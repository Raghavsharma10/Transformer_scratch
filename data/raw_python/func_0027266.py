def assert_unordered_list_eq(expected, actual, message=None):
    """Raises an AssertionError if the objects contained
    in expected are not equal to the objects contained
    in actual  without regard to their order.

    This takes quadratic time in the umber of elements in actual; don't use it for very long lists.

    """
    missing_in_actual = []
    missing_in_expected = list(actual)
    for x in expected:
        try:
            missing_in_expected.remove(x)
        except ValueError:
            missing_in_actual.append(x)

    if missing_in_actual or missing_in_expected:
        if not message:
            message = (
                "%r not equal to %r; missing items: %r in expected, %r in actual."
                % (expected, actual, missing_in_expected, missing_in_actual)
            )
        assert False, message