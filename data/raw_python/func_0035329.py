def number_of_bytes_to_modify(buf_len, fuzz_factor):
    """Calculate number of bytes to modify.

    :param buf_len: len of data buffer to fuzz.
    :param fuzz_factor: degree of fuzzing.
    :return: number of bytes to change.
    """
    return random.randrange(math.ceil((float(buf_len) / fuzz_factor))) + 1