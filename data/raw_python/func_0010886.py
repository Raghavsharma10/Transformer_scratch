def encode_array(input_array, codec, param):
    """Encode the array using the method and then add the header to this array.

    :param input_array: the array to be encoded
    :param codec: the integer index of the codec to use
    :param param: the integer parameter to use in the function
    :return an array with the header added to the fornt"""
    return add_header(codec_dict[codec].encode(input_array, param), codec, len(input_array), param)