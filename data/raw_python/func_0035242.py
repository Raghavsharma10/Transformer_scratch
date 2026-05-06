def step_impl07(context, len_list):
    """Check assertions.

    :param len_list: expected number of variants.
    :param context: test context.
    """
    assert len(context.fuzzed_string_list) == len_list
    for fuzzed_string in context.fuzzed_string_list:
        assert len(context.seed) == len(fuzzed_string)
        count = number_of_modified_bytes(context.seed, fuzzed_string)
        assert count >= 0