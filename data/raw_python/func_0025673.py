def copyFile(input, output, replace=None):
    """Copy a file whole from input to output."""

    _found = findFile(output)
    if not _found or (_found and replace):
        shutil.copy2(input, output)