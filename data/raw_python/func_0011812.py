def _get_rar_version(xfile):
    """Check quickly whether file is rar archive.
    """
    buf = xfile.read(len(RAR5_ID))
    if buf.startswith(RAR_ID):
        return 3
    elif buf.startswith(RAR5_ID):
        xfile.read(1)
        return 5
    return 0