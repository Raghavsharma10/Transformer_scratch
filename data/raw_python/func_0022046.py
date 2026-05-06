def compute_gap(start, end, time_limit):
    """
    Compute a gap that seems reasonable, considering natural time units and limit.
    # TODO: make it to be reasonable.
    # TODO: make it to be small unit of time sensitive.
    :param start: datetime
    :param end: datetime
    :param time_limit: gaps count
    :return: solr's format duration.
    """
    if is_range_common_era(start, end):
        duration = end.get("parsed_datetime") - start.get("parsed_datetime")
        unit = int(math.ceil(duration.days / float(time_limit)))
        return "+{0}DAYS".format(unit)
    else:
        # at the moment can not do maths with BCE dates.
        # those dates are relatively big, so 100 years are reasonable in those cases.
        # TODO: calculate duration on those cases.
        return "+100YEARS"