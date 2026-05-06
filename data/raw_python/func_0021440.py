def calc_hexversion(major=0, minor=0, micro=0, releaselevel='dev', serial=0):
    """Calculate the hexadecimal version number from the tuple version_info:

    :param major: integer
    :param minor: integer
    :param micro: integer
    :param relev: integer or string
    :param serial: integer
    :return: integerm always increasing with revision numbers
    """
    try:
        releaselevel = int(releaselevel)
    except ValueError:
        releaselevel = RELEASE_LEVEL_VALUE.get(releaselevel, 0)

    hex_version = int(serial)
    hex_version |= releaselevel * 1 << 4
    hex_version |= int(micro) * 1 << 8
    hex_version |= int(minor) * 1 << 16
    hex_version |= int(major) * 1 << 24
    return hex_version