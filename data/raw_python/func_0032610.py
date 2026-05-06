def unpackfile(filename, destdir):
    """Unpack a mar or exe into destdir"""
    if filename.endswith(".mar"):
        return unpackmar(filename, destdir)
    elif filename.endswith(".exe"):
        return unpackexe(filename, destdir)
    elif filename.endswith(".tar") or filename.endswith(".tar.gz") \
            or filename.endswith(".tgz"):
        return unpacktar(filename, destdir)
    else:
        raise ValueError("Unknown file type: %s" % filename)