def parse_tz(tz):
    """Parse a timezone specification in the [+|-]HHMM format.

    :return: the timezone offset in seconds.
    """
    # from git_repository.py in bzr-git
    sign_byte = tz[0:1]
    # in python 3 b'+006'[0] would return an integer,
    # but b'+006'[0:1] return a new bytes string.
    if sign_byte not in (b'+', b'-'):
        raise ValueError(tz)

    sign = {b'+': +1, b'-': -1}[sign_byte]
    hours = int(tz[1:-2])
    minutes = int(tz[-2:])

    return sign * 60 * (60 * hours + minutes)