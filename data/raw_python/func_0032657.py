def maskInMicrolensRegion(ch, col, row, padding=0):
    """Is a target in the K2C9 superstamp, including padding?

    This function is identical to pixelInMicrolensRegion, except it takes
    the extra `padding` argument. The coordinate must be within the K2C9
    superstamp by at least `padding` number of pixels on either side of the
    coordinate.  (Note that this function does not check whether something is
    close to the CCD boundaries, it only checks whether something is close
    to the edge of stamp.)
    """
    if padding == 0:
        return pixelInMicrolensRegion(ch, col, row)

    combinations = [[col - padding, row],
                    [col + padding, row],
                    [col, row - padding],
                    [col, row + padding]]
    for col, row in combinations:
        # Science pixels occupy columns 12 - 1111, rows 20 - 1043
        if col < 12:
            col = 12
        if col > 1111:
            col = 1111
        if row < 20:
            row = 20
        if row > 1043:
            row = 1043
        if not pixelInMicrolensRegion(ch, col, row):
            return False
    return True