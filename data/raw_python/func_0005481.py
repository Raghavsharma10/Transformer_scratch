def chars_in(bits, keyspace):
    """ ..
        log2(keyspace^x_chars) = bits
        log(keyspace^x_chars) = log(2) * bits
        exp(log(keyspace^x_chars)) = exp(log(2) * bits)
        x_chars = log(exp(log(2) * bits)) / log(keyspace)
        ..
        -> (#int) number of characters in @bits of entropy given the @keyspace
    """
    keyspace = len(keyspace)
    if keyspace < 2:
        raise ValueError("Keyspace size must be >1")
    bits_per_cycle = 512
    if bits > bits_per_cycle:
        chars = 0
        bits_processed = 0
        cycles = ceil(bits / bits_per_cycle)
        for _ in range(int(cycles)):
            if bits_processed + bits_per_cycle > bits:
                bits_per_cycle = bits - bits_processed
            chars += calc_chars_in(bits_per_cycle, keyspace)
            bits_processed += bits_per_cycle
    else:
        chars = calc_chars_in(bits, keyspace)
    return abs(chars)