def get_channel(cell):
    """ Gets the channel of a network / cell.
    @param string cell
        A network / cell from iwlist scan.

    @return string
        The channel of the network.
    """

    channel = matching_line(cell, "Channel:")
    if channel:
        return channel
    frequency = matching_line(cell, "Frequency:")
    channel = re.sub(r".*\(Channel\s(\d{1,3})\).*", r"\1", frequency)
    return channel