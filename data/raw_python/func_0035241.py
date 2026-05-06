def step_impl06(context, count):
    """Execute fuzzer.

    :param count: number of string variants to generate.
    :param context: test context.
    """
    fuzz_factor = 11
    context.fuzzed_string_list = fuzz_string(context.seed, count, fuzz_factor)