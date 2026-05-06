def parse_interval_records(interval_record, interval_date, interval, uom,
                           quality_method) -> List[Reading]:
    """ Convert interval values into tuples with datetime
    """
    interval_delta = timedelta(minutes=interval)
    return [
        Reading(
            t_start=interval_date + (i * interval_delta),
            t_end=interval_date + (i * interval_delta) + interval_delta,
            read_value=parse_reading(val),
            uom=uom,
            quality_method=quality_method,
            event_code="",  # event is unknown at time of reading
            event_desc="",  # event is unknown at time of reading
            read_start=None,
            read_end=None  # No before and after readings for intervals
        ) for i, val in enumerate(interval_record)
    ]