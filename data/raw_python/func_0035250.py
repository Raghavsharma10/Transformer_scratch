def number_of_modified_bytes(buf, fuzzed_buf):
    """Determine the number of differing bytes.

    :param buf: original buffer.
    :param fuzzed_buf: fuzzed buffer.
    :return: number of different bytes.
    :rtype: int
    """
    count = 0
    for idx, b in enumerate(buf):
        if b != fuzzed_buf[idx]:
            count += 1
    return count