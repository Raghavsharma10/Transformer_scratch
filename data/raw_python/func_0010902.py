def add_header(input_array, codec, length, param):
    """Add the header to the appropriate array.
    :param the encoded array to add the header to
    :param the codec being used
    :param the length of the decoded array
    :param the parameter to add to the header
    :return the prepended encoded byte array"""
    return struct.pack(mmtf.utils.constants.NUM_DICT[4], codec) + \
           struct.pack(mmtf.utils.constants.NUM_DICT[4], length) + \
           struct.pack(mmtf.utils.constants.NUM_DICT[4], param) + input_array