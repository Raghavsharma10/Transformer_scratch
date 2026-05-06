def assert_dict_eq(expected, actual, number_tolerance=None, dict_path=[]):
    """Asserts that two dictionaries are equal, producing a custom message if they are not."""
    assert_is_instance(expected, dict)
    assert_is_instance(actual, dict)

    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    assert expected_keys <= actual_keys, "Actual dict at %s is missing keys: %r" % (
        _dict_path_string(dict_path),
        expected_keys - actual_keys,
    )
    assert actual_keys <= expected_keys, "Actual dict at %s has extra keys: %r" % (
        _dict_path_string(dict_path),
        actual_keys - expected_keys,
    )

    for k in expected_keys:
        key_path = dict_path + [k]
        assert_is_instance(
            actual[k],
            type(expected[k]),
            extra="Types don't match for %s" % _dict_path_string(key_path),
        )
        assert_is_instance(
            expected[k],
            type(actual[k]),
            extra="Types don't match for %s" % _dict_path_string(key_path),
        )

        if isinstance(actual[k], dict):
            assert_dict_eq(
                expected[k],
                actual[k],
                number_tolerance=number_tolerance,
                dict_path=key_path,
            )
        elif isinstance(actual[k], _number_types):
            assert_eq(
                expected[k],
                actual[k],
                extra="Value doesn't match for %s" % _dict_path_string(key_path),
                tolerance=number_tolerance,
            )
        else:
            assert_eq(
                expected[k],
                actual[k],
                extra="Value doesn't match for %s" % _dict_path_string(key_path),
            )