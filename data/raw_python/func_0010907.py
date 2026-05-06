def recursive_index_encode(int_array, max=32767, min=-32768):
    """Pack an integer array using recursive indexing.

    :param int_array: the input array of integers
    :param max: the maximum integer size
    :param min: the minimum integer size
    :return the array of integers after recursive index encoding"""
    out_arr = []
    for curr in int_array:
        if curr >= 0 :
            while curr >= max:
                out_arr.append(max)
                curr -=  max
        else:
            while curr <= min:
                out_arr.append(min)
                curr += int(math.fabs(min))
        out_arr.append(curr)
    return out_arr