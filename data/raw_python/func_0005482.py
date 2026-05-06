def bits_in(length, keyspace):
    """ |log2(keyspace^length) = bits|
        -> (#float) number of bits of entropy in @length of characters for
            a given a @keyspace
    """
    keyspace = len(keyspace)
    length_per_cycle = 64
    if length > length_per_cycle:
        bits = 0
        length_processed = 0
        cycles = ceil(length / length_per_cycle)
        for _ in range(int(cycles)):
            if length_processed + length_per_cycle > length:
                length_per_cycle = length - length_processed
            bits += calc_bits_in(length_per_cycle, keyspace)
            length_processed += length_per_cycle
    else:
        bits = calc_bits_in(length, keyspace)
    return float(abs(bits))