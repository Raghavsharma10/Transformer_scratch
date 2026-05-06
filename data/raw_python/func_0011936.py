def parse_source_file(source_file):
    """Parses a source file thing and returns the file name

    Example:

    >>> parse_file('hg:hg.mozilla.org/releases/mozilla-esr52:js/src/jit/MIR.h:755067c14b06')
    'js/src/jit/MIR.h'

    :arg str source_file: the source file ("file") from a stack frame

    :returns: the filename or ``None`` if it couldn't determine one

    """
    if not source_file:
        return None

    vcsinfo = source_file.split(':')
    if len(vcsinfo) == 4:
        # These are repositories or cloud file systems (e.g. hg, git, s3)
        vcstype, root, vcs_source_file, revision = vcsinfo
        return vcs_source_file

    if len(vcsinfo) == 2:
        # These are directories on someone's Windows computer and vcstype is a
        # file system (e.g. "c:", "d:", "f:")
        vcstype, vcs_source_file = vcsinfo
        return vcs_source_file

    if source_file.startswith('/'):
        # These are directories on OSX or Linux
        return source_file

    # We have no idea what this is, so return None
    return None