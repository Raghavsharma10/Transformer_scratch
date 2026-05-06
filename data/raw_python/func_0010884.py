def delta_encode(in_array):
    """A function to delta decode an int array.

    :param in_array: the inut array to be delta encoded
    :return the encoded integer array"""
    if(len(in_array)==0):
        return []
    curr_ans = in_array[0]
    out_array = [curr_ans]
    for in_int in in_array[1:]:
        out_array.append(in_int-curr_ans)
        curr_ans = in_int
    return out_array