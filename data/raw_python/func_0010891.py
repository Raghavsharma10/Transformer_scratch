def recursive_index_decode(int_array, max=32767, min=-32768):
    """Unpack an array of integers using recursive indexing.

    :param int_array: the input array of integers
    :param max: the maximum integer size
    :param min: the minimum integer size
    :return the array of integers after recursive index decoding"""
    out_arr = []
    decoded_val = 0
    for item in int_array.tolist():
        if item==max or item==min:
            decoded_val += item
        else:
            decoded_val += item
            out_arr.append(decoded_val)
            decoded_val = 0
    return numpy.asarray(out_arr,dtype=numpy.int32)