def recursive_index_decode(int_array, max=32767, min=-32768):
    """Unpack an array of integers using recursive indexing.
    :param int_array: the input array of integers
    :param max: the maximum integer size
    :param min: the minimum integer size
    :return the array of integers after recursive index decoding"""
    out_arr = []
    encoded_ind = 0
    while encoded_ind < len(int_array):
        decoded_val = 0
        while int_array[encoded_ind]==max or int_array[encoded_ind]==min:
            decoded_val += int_array[encoded_ind]
            encoded_ind+=1
            if int_array[encoded_ind]==0:
                break
        decoded_val += int_array[encoded_ind]
        encoded_ind+=1
        out_arr.append(decoded_val)
    return out_arr