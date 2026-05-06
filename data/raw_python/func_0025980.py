def rglob(root, pattern):
    """ Same thing as glob.glob, but recursively checks subdirs. """
    # Thanks to Alex Martelli for basics on Stack Overflow
    retlist = []
    if None not in (pattern, root):
        for base, dirs, files in os.walk(root):
            goodfiles = fnmatch.filter(files, pattern)
            retlist.extend(os.path.join(base, f) for f in goodfiles)
    return retlist