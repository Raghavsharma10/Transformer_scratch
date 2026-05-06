def read_files(filenames):
    """Read a file into memory."""
    if isinstance(filenames, list):
        for filename in filenames:
            with open(filename, 'r') as infile:
                return infile.read()
    else:
        with open(filenames, 'r') as infile:
            return infile.read()