def get_signal_level(cell):
    """ Gets the signal level of a network / cell.
    @param string cell
        A network / cell from iwlist scan.

    @return string
        The signal level of the network.
    """

    signal = matching_line(cell, "Signal level=")
    if signal is None:
      return ""
    signal = signal.split("=")[1].split("/")
    if len(signal) == 2:
        return str(int(round(float(signal[0]) / float(signal[1]) * 100)))
    elif len(signal) == 1:
        return signal[0].split(' ')[0]
    else:
        return ""