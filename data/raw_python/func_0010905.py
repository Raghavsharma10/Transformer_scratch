def decode_chain_list(in_bytes):
    """Convert a list of bytes to a list of strings. Each string is of length mmtf.CHAIN_LEN

    :param in_bytes: the input bytes
    :return the decoded list of strings"""
    tot_strings = len(in_bytes) // mmtf.utils.constants.CHAIN_LEN
    out_strings = []
    for i in range(tot_strings):
        out_s = in_bytes[i * mmtf.utils.constants.CHAIN_LEN:i * mmtf.utils.constants.CHAIN_LEN + mmtf.utils.constants.CHAIN_LEN]
        out_strings.append(out_s.decode("ascii").strip(mmtf.utils.constants.NULL_BYTE))
    return out_strings