def step_impl(context, max_modified):
    """Check assertions.

    :param max_modified: maximum expected number of modifications.
    :param context: test context.
    """
    assert len(context.buf) == len(context.fuzzed_buf)
    count = number_of_modified_bytes(context.buf, context.fuzzed_buf)
    assert count >= 0
    assert count <= max_modified