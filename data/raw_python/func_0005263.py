def parse_date_range_arguments(options: dict, default_range='last_month') -> (datetime, datetime, list):
    """
    :param options:
    :param default_range: Default datetime range to return if no other selected
    :return: begin, end, [(begin1,end1), (begin2,end2), ...]
    """
    begin, end = get_date_range_by_name(default_range)
    for range_name in TIME_RANGE_NAMES:
        if options.get(range_name):
            begin, end = get_date_range_by_name(range_name)
    if options.get('begin'):
        t = parse(options['begin'], default=datetime(2000, 1, 1))
        begin = pytz.utc.localize(t)
        end = now()
    if options.get('end'):
        end = pytz.utc.localize(parse(options['end'], default=datetime(2000, 1, 1)))

    step_type = None
    after_end = end
    for step_name in TIME_STEP_NAMES:
        if options.get(step_name):
            step_type = getattr(rrule, step_name.upper())
            if rrule.DAILY == step_type:
                after_end += timedelta(days=1)
            if rrule.WEEKLY == step_type:
                after_end += timedelta(days=7)
            if rrule.MONTHLY == step_type:
                after_end += timedelta(days=31)
    steps = None
    if step_type:
        begins = [t for t in rrule.rrule(step_type, dtstart=begin, until=after_end)]
        steps = [(begins[i], begins[i+1]) for i in range(len(begins)-1)]
    if steps is None:
        steps = [(begin, end)]
    return begin, end, steps