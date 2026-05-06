def decode_chain_list(in_bytes):
    """Convert a list of bytes to a list of strings. Each string is of length mmtf.CHAIN_LEN

    :param in_bytes: the input bytes
    :return the decoded list of strings"""
    bstrings = numpy.frombuffer(in_bytes, numpy.dtype('S' + str(mmtf.utils.constants.CHAIN_LEN)))
    return [s.decode("ascii").strip(mmtf.utils.constants.NULL_BYTE) for s in bstrings]