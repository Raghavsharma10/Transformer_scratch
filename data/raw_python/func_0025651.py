def countinputs(inputlist):
    """
    Determine the number of inputfiles provided by the user and the
    number of those files that are association tables

    Parameters
    ----------
    inputlist   : string
        the user input

    Returns
    -------
    numInputs: int
        number of inputs provided by the user
    numASNfiles: int
        number of association files provided as input
    """

    # Initialize return values
    numInputs = 0
    numASNfiles = 0

    # User irafglob to count the number of inputfiles
    files = irafglob(inputlist, atfile=None)

    # Use the "len" ufunc to count the number of entries in the list
    numInputs = len(files)

    # Loop over the list and see if any of the entries are association files
    for file in files:
        if (checkASN(file) == True):
            numASNfiles += 1

    return numInputs,numASNfiles