def convert_ints_to_bytes(in_ints, num):
    """Convert an integer array into a byte arrays. The number of bytes forming an integer
    is defined by num

    :param in_ints: the input integers
    :param num: the number of bytes per int
    :return the integer array"""
    out_bytes= b""
    for val in in_ints:
        out_bytes+=struct.pack(mmtf.utils.constants.NUM_DICT[num], val)
    return out_bytes