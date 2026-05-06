def get_noise_level(cell):
    """ Gets the noise level of a network / cell.
    @param string cell
        A network / cell from iwlist scan.

    @return string
        The noise level of the network.
    """

    noise = matching_line(cell, "Noise level=")
    if noise is None:
        return ""
    noise = noise.split("=")[1]
    return noise.split(' ')[0]