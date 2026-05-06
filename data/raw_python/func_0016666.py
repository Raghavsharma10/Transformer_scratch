def _decode_byte(byte):
    """Helper for decode_qp, turns a single byte into a list of bits.

    Args:
        byte: byte to be decoded

    Returns:
        list of bits corresponding to byte
    """
    bits = []
    for _ in range(8):
        bits.append(byte & 1)
        byte >>= 1
    return bits