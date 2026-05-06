def _mic_required(target_info):
    """
    Checks the MsvAvFlags field of the supplied TargetInfo structure to determine in the MIC flags is set
    :param target_info: The TargetInfo structure to check
    :return: a boolean value indicating that the MIC flag is set
    """
    if target_info is not None and target_info[TargetInfo.NTLMSSP_AV_FLAGS] is not None:
        flags = struct.unpack('<I', target_info[TargetInfo.NTLMSSP_AV_FLAGS][1])[0]
        return bool(flags & 0x00000002)