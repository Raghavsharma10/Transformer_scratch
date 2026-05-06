def assert_not_in(obj, seq, message=None, extra=None):
    """Raises an AssertionError if obj is in iter."""
    # for very long strings, provide a truncated error
    if isinstance(seq, six.string_types) and obj in seq and len(seq) > 200:
        index = seq.find(obj)
        start_index = index - 50
        if start_index > 0:
            truncated = "(truncated) ..."
        else:
            truncated = ""
            start_index = 0
        end_index = index + len(obj) + 50
        truncated += seq[start_index:end_index]
        if end_index < len(seq):
            truncated += "... (truncated)"
        assert False, _assert_fail_message(message, obj, truncated, "is in", extra)
    assert obj not in seq, _assert_fail_message(message, obj, seq, "is in", extra)