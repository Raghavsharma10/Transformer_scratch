def decode_array(input_array):
    """Parse the header of an input byte array and then decode using the input array,
    the codec and the appropirate parameter.

    :param input_array: the array to be decoded
    :return the decoded array"""
    codec, length, param, input_array = parse_header(input_array)
    return codec_dict[codec].decode(input_array, param)