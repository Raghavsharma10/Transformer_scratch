def fuzzer(buffer, fuzz_factor=101):
    """Fuzz given buffer.

    Take a buffer of bytes, create a copy, and replace some bytes
    with random values. Number of bytes to modify depends on fuzz_factor.
    This code is taken from Charlie Miller's fuzzer code.

    :param buffer: the data to fuzz.
    :type buffer: byte array
    :param fuzz_factor: degree of fuzzing.
    :type fuzz_factor: int
    :return: fuzzed buffer.
    :rtype: byte array
    """
    buf = deepcopy(buffer)
    num_writes = number_of_bytes_to_modify(len(buf), fuzz_factor)
    for _ in range(num_writes):
        random_byte = random.randrange(256)
        random_position = random.randrange(len(buf))
        buf[random_position] = random_byte
    return buf