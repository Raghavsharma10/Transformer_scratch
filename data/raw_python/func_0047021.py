def check_read_offsets(read_offsets):
    """Check if read offsets are valid (positive)."""
    for read_offset in read_offsets:
        if read_offset < 0:
            msg = 'Read offset must be 0 or greater'
            log.error(msg)
            raise ArgumentError(msg)