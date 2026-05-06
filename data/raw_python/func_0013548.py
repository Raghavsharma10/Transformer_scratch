def getNameFromPath(filePath):
    """
    Returns the filename of the specified path without its extensions.
    This is usually how we derive the default name for a given object.
    """
    if len(filePath) == 0:
        raise ValueError("Cannot have empty path for name")
    fileName = os.path.split(os.path.normpath(filePath))[1]
    # We need to handle things like .fa.gz, so we can't use
    # os.path.splitext
    ret = fileName.split(".")[0]
    assert ret != ""
    return ret