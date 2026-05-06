def parse_0101(v):
    """
    Parses the DTC status and returns two elements.
    https://en.wikipedia.org/wiki/OBD-II_PIDs#Mode_1_PID_01
    :param v:
    :return bool, int:
    """
    tv = trim_obd_value(v)
    mil_status = None  # type: bool
    num_dtc = None  # type: int

    try:
        byte_a = int(v[:2], 16)
        mil_status = byte_a / 0xF >= 1
        num_dtc = mil_status % 0xF
    except ValueError:
        mil_status = None
        num_dtc = None

    return mil_status, num_dtc