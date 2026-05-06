def parse_ISO8601(time_gap):
    """
    P1D to (1, ("DAYS", isodate.Duration(days=1)).
    P1Y to (1, ("YEARS", isodate.Duration(years=1)).
    :param time_gap: ISO8601 string.
    :return: tuple with quantity and unit of time.
    """
    matcher = None

    if time_gap.count("T"):
        units = {
            "H": ("HOURS", isodate.Duration(hours=1)),
            "M": ("MINUTES", isodate.Duration(minutes=1)),
            "S": ("SECONDS", isodate.Duration(seconds=1))
        }
        matcher = re.search("PT(\d+)([HMS])", time_gap)
        if matcher:
            quantity = int(matcher.group(1))
            unit = matcher.group(2)
            return quantity, units.get(unit)
        else:
            raise Exception("Does not match the pattern: {}".format(time_gap))
    else:
        units = {
            "Y": ("YEARS", isodate.Duration(years=1)),
            "M": ("MONTHS", isodate.Duration(months=1)),
            "W": ("WEEKS", isodate.Duration(weeks=1)),
            "D": ("DAYS", isodate.Duration(days=1))
        }
        matcher = re.search("P(\d+)([YMWD])", time_gap)
        if matcher:
            quantity = int(matcher.group(1))
            unit = matcher.group(2)
        else:
            raise Exception("Does not match the pattern: {}".format(time_gap))

    return quantity, units.get(unit)