def check_file_version(file):
    """Check the ARF version attribute of file for compatibility.

    Raises DeprecationWarning for backwards-incompatible files, FutureWarning
    for (potentially) forwards-incompatible files, and UserWarning for files
    that may not have been created by an ARF library.

    Returns the version for the file

    """
    from distutils.version import StrictVersion as Version
    try:
        ver = file.attrs.get('arf_version', None)
        if ver is None:
            ver = file.attrs['arf_library_version']
    except KeyError:
        raise UserWarning(
            "Unable to determine ARF version for {0.filename};"
            "created by another program?".format(file))
    try:
        # if the attribute is stored as a string, it's ascii-encoded
        ver = ver.decode("ascii")
    except (LookupError, AttributeError):
        pass
    # should be backwards compatible after 1.1
    file_version = Version(ver)
    if file_version < Version('1.1'):
        raise DeprecationWarning(
            "ARF library {} may have trouble reading file "
            "version {} (< 1.1)".format(version, file_version))
    elif file_version >= Version('3.0'):
        raise FutureWarning(
            "ARF library {} may be incompatible with file "
            "version {} (>= 3.0)".format(version, file_version))
    return file_version