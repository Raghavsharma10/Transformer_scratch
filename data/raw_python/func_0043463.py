def parse_0103(v):
    """
    Parses the fuel system status and returns an array with two elements (one for
    each fuel system).
    The returned values are converted to decimal integers and returned as is.
    The fuel system values are described here:
    https://en.wikipedia.org/wiki/OBD-II_PIDs#Mode_1_PID_03

    1  Open loop due to insufficient engine temperature

    2  Closed loop, using oxygen sensor feedback to determine fuel mix

    4  Open loop due to engine load OR fuel cut due to deceleration

    8  Open loop due to system failure

    16 Closed loop, using at least one oxygen sensor but there is a fault in the feedback system

    :param str v: e.g. "41030100"
    :return int, int:
    """
    tv = trim_obd_value(v)  # trimmed value
    status_1, status_2 = None, None
    try:
        status_1 = int(v[:2], 16)
    except ValueError:
        status_1 = None

    try:
        status_2 = int(v[2:4], 16)
    except ValueError:
        status_2 = None

    return status_1, status_2