def update_reading_events(readings, event_record):
    """ Updates readings from a 300 row to reflect any events found in a
        subsequent 400 row
    """
    # event intervals are 1-indexed
    for i in range(event_record.start_interval - 1, event_record.end_interval):
        readings[i] = Reading(
            t_start=readings[i].t_start,
            t_end=readings[i].t_end,
            read_value=readings[i].read_value,
            uom=readings[i].uom,
            quality_method=event_record.quality_method,
            event_code=event_record.reason_code,
            event_desc=event_record.reason_description,
            read_start=readings[i].read_start,
            read_end=readings[i].read_end)
    return readings