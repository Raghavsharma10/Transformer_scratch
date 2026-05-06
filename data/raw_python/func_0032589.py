def convertPath(srcpath, dstdir):
    """Given `srcpath`, return a corresponding path within `dstdir`"""
    bits = srcpath.split("/")
    bits.pop(0)
    # Strip out leading 'unsigned' from paths like unsigned/update/win32/...
    if bits[0] == 'unsigned':
        bits.pop(0)
    return os.path.join(dstdir, *bits)