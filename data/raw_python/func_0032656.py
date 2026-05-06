def pixelInMicrolensRegion(ch, col, row):
    """Returns `True` if the given pixel falls inside the K2C9 superstamp.

    The superstamp is used for microlensing experiment and is an almost
    contiguous area of 2.8e6 pixels.
    """
    # First try the superstamp
    try:
        vertices_col = SUPERSTAMP["channels"][str(int(ch))]["vertices_col"]
        vertices_row = SUPERSTAMP["channels"][str(int(ch))]["vertices_row"]
        # The point is in one of 5 channels which constitute the superstamp
        # so check if it falls inside the polygon for this channel
        if isPointInsidePolygon(col, row, vertices_col, vertices_row):
            return True
    except KeyError:  # Channel does not appear in file
        pass

    # Then try the late target masks
    for mask in LATE_TARGETS["masks"]:
        if mask["channel"] == ch:
            vertices_col = mask["vertices_col"]
            vertices_row = mask["vertices_row"]
            if isPointInsidePolygon(col, row, vertices_col, vertices_row):
                return True

    return False