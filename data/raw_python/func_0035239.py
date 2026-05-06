def step_impl03(context):
    """Check assertions.

    :param context: test context.
    """
    assert len(context.buf) == len(context.fuzzed_buf)
    count = number_of_modified_bytes(context.buf, context.fuzzed_buf)
    assert count < 3
    assert count >= 0