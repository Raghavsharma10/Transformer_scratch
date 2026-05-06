def xor_bytes(b1: bytes, b2: bytes) -> bytearray:
    """
    Apply XOR operation on two bytes arguments

    :param b1: First bytes argument
    :param b2: Second bytes argument
    :rtype bytearray:
    """
    result = bytearray()
    for i1, i2 in zip(b1, b2):
        result.append(i1 ^ i2)
    return result