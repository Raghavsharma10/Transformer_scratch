def packfile(filename, srcdir):
    """Package up srcdir into filename, archived with 7z for exes or mar for
    mar files"""
    if filename.endswith(".mar"):
        return packmar(filename, srcdir)
    elif filename.endswith(".exe"):
        return packexe(filename, srcdir)
    elif filename.endswith(".tar"):
        return tar_dir(filename, srcdir)
    else:
        raise ValueError("Unknown file type: %s" % filename)