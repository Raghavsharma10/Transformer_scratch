def convert_bytes_to_ints(in_bytes, num):
    """Convert a byte array into an integer array. The number of bytes forming an integer
    is defined by num
    :param in_bytes: the input bytes
    :param num: the number of bytes per int
    :return the integer array"""
    out_arr = []
    for i in range(len(in_bytes)//num):
        val = in_bytes[i * num:i * num + num]
        unpacked = struct.unpack(mmtf.utils.constants.NUM_DICT[num], val)
        out_arr.append(unpacked[0])
    return out_arr