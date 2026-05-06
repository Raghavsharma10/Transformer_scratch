def findNextFile(folder='.',
                 prefix=None,
                 suffix=None,
                 fnameGen=None,
                 base=0,
                 maxattempts=10):
    """Finds the next available file-name in a sequence.

    This function will create a file of zero size and will return the path to
    it to the caller. No files which exist will be altered in this operation
    and concurrent executions of this function will return separate files. In
    case of conflict, the function will attempt to generate a new file name up
    to maxattempts number of times before failing.

    The sequence will start from the base argument (default: 0).

    If used with the prefix/suffix, it will look for the next file in the
    sequence ignoring any gaps. Hence, if the files "a.0.txt" and "a.3.txt"
    exist, then the next file returned will be "a.4.txt" when called with
    prefix="a." and suffix=".txt".

    In case fnameGen is provided, the first generated filename which does not
    exist will be created and its path will be returned. Hence, if the files
    "a.0.txt" and "a.3.txt" exist, then the next file returned will be
    "a.1.txt" when called with fnameGen = lambda x : "a." + str(x) + ".txt"

    Args:
        folder      - string which has path to the folder where the file should
                      be created (default: '.')
        prefix      - prefix of the file to be generated (default: '')
        suffix      - suffix of the file to be generated (default: '')
        fnameGen    - function which generates the filenames given a number as
                      input (default: None)
        base        - the first index to count (default: 0)
        maxattempts - number of attempts to create the file before failing with
                      OSError (default: 10)

    Returns:
        Path of the file which follows the provided pattern and can be opened
        for writing.

    Raises:
        RuntimeError - If an incorrect combination of arguments is provided.
        OSError      - If is unable to create a file (wrong path, drive full,
                       illegal character in filename, etc.).
    """
    expFolder = _os.path.expanduser(_os.path.expandvars(folder))
    return _findNextFile(expFolder, prefix, suffix, fnameGen,
                         base, maxattempts, 0)