def encode_chain_list(in_strings):
    """Convert a list of strings to a list of byte arrays.

    :param in_strings: the input strings
    :return the encoded list of byte arrays"""
    out_bytes = b""
    for in_s in in_strings:
        out_bytes+=in_s.encode('ascii')
        for i in range(mmtf.utils.constants.CHAIN_LEN -len(in_s)):
            out_bytes+= mmtf.utils.constants.NULL_BYTE.encode('ascii')
    return out_bytes