def parse_010c(v) -> int:
    """
    Parses Engine RPM and returns it in [RPM] as a float from 0 - 16383.75
    :param str v:
    :return int:
    """
    try:
        val = int(trim_obd_value(v), 16)
        return int(val / 4)
    except ValueError:
        return None