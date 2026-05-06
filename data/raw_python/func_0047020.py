def check_read_lengths(ribo_file, read_lengths):
    """Check if read lengths are valid (positive). """
    # check if there are any valid read lengths to check i.e., not equal to 0
    valid_lengths = list(set(read_lengths))
    # if read length is 0, all read lengths are requested so we skip further
    # checks.
    if len(valid_lengths) == 1 and valid_lengths[0] == 0:
        return
    for read_length in valid_lengths:
        if read_length < 0:
            msg = 'Read length must be a positive value'
            log.error(msg)
            raise ArgumentError(msg)