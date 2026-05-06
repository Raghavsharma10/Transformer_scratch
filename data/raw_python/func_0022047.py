def gap_to_sorl(time_gap):
    """
    P1D to +1DAY
    :param time_gap:
    :return: solr's format duration.
    """
    quantity, unit = parse_ISO8601(time_gap)
    if unit[0] == "WEEKS":
        return "+{0}DAYS".format(quantity * 7)
    else:
        return "+{0}{1}".format(quantity, unit[0])