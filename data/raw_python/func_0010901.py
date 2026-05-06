def parse_header(input_array):
    """Parse the header and return it along with the input array minus the header.
    :param input_array the array to parse
    :return the codec, the length of the decoded array, the parameter and the remainder
    of the array"""
    codec = struct.unpack(mmtf.utils.constants.NUM_DICT[4], input_array[0:4])[0]
    length = struct.unpack(mmtf.utils.constants.NUM_DICT[4], input_array[4:8])[0]
    param = struct.unpack(mmtf.utils.constants.NUM_DICT[4], input_array[8:12])[0]
    return codec,length,param,input_array[12:]