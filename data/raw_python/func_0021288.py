def get_quality(cell):
    """ Gets the quality of a network / cell.
    @param string cell
        A network / cell from iwlist scan.

    @return string
        The quality of the network.
    """

    quality = matching_line(cell, "Quality=")
    if quality is None:
        return ""
    quality = quality.split()[0].split("/")
    quality = matching_line(cell, "Quality=").split()[0].split("/")
    return str(int(round(float(quality[0]) / float(quality[1]) * 100)))