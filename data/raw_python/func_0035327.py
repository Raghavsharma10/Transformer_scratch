def fuzz_string(seed_str, runs=100, fuzz_factor=50):
    """A random fuzzer for a simulated text viewer application.

    It takes a string as seed and generates <runs> variant of it.

    :param seed_str: the string to use as seed for fuzzing.
    :param runs: number of fuzzed variants to supply.
    :param fuzz_factor: degree of fuzzing = 1 / fuzz_factor.
    :return: list of fuzzed variants of seed_str.
    :rtype: [str]
    """
    buf = bytearray(seed_str, encoding="utf8")
    variants = []
    for _ in range(runs):
        fuzzed = fuzzer(buf, fuzz_factor)
        variants.append(''.join([chr(b) for b in fuzzed]))
    logger().info('Fuzzed strings: {}'.format(variants))
    return variants