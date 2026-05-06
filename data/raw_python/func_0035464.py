def _make_win(n, mono=False):
    """ Generate a window for a given length.

    :param n: an integer for the length of the window.
    :param mono: True for a mono window, False for a stereo window.
    :return: an numpy array containing the window value.
    """

    if mono:
        win = np.hanning(n) + 0.00001
    else:
        win = np.array([np.hanning(n) + 0.00001, np.hanning(n) + 0.00001])
    win = np.transpose(win)
    return win