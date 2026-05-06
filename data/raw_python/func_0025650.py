def checkASN(filename):
    """
    Determine if the filename provided to the function belongs to
    an association.

    Parameters
    ----------
    filename: string

    Returns
    -------
    validASN  : boolean value

    """
    # Extract the file extn type:
    extnType = filename[filename.rfind('_')+1:filename.rfind('.')]

    # Determine if this extn name is valid for an assocation file
    if isValidAssocExtn(extnType):
        return True
    else:
        return False