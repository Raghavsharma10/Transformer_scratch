def parse_0134_013b(v):
    """
    Parses the O2 Sensor Value (0134 - 013B) and returns two values parsed from it:
    1. Fuel-Air Equivalence [Ratio] as a float from 0 - 2
    2. Current in [mA] as a float from -128 - 128
    :param str v:
    :return tuple of float, float:
    """
    try:
        trim_val = trim_obd_value(v)
        val_ab = int(trim_val[0:2], 16)
        val_cd = int(trim_val[2:4], 16)
        return (2 / 65536) * val_ab, val_cd - 128
    except ValueError:
        return None, None