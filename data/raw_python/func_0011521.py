def bits_to_bytes(bits):
    """Convert the bit list into bytes. (Assumes bits is a list
    whose length is a multiple of 8)
    """
    if len(bits) % 8 != 0:
        raise Exception("num bits must be multiple of 8")

    res = ""

    for x in six.moves.range(0, len(bits), 8):
        byte_bits = bits[x:x+8]
        byte_val = int(''.join(map(str, byte_bits)), 2)
        res += chr(byte_val)

    return utils.binary(res)