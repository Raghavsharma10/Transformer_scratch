def is_open(location, now=None):
    """
    Is the company currently open? Pass "now" to test with a specific
    timestamp. Can be used stand-alone or as a helper.
    """
    if now is None:
        now = get_now()

    if has_closing_rule_for_now(location):
        return False

    now_time = datetime.time(now.hour, now.minute, now.second)

    if location:
        ohs = OpeningHours.objects.filter(company=location)
    else:
        ohs = Company.objects.first().openinghours_set.all()
    for oh in ohs:
        is_open = False
        # start and end is on the same day
        if (oh.weekday == now.isoweekday() and
                oh.from_hour <= now_time and
                now_time <= oh.to_hour):
            is_open = oh

        # start and end are not on the same day and we test on the start day
        if (oh.weekday == now.isoweekday() and
                oh.from_hour <= now_time and
                ((oh.to_hour < oh.from_hour) and
                    (now_time < datetime.time(23, 59, 59)))):
            is_open = oh

        # start and end are not on the same day and we test on the end day
        if (oh.weekday == (now.isoweekday() - 1) % 7 and
                oh.from_hour >= now_time and
                oh.to_hour >= now_time and
                oh.to_hour < oh.from_hour):
            is_open = oh
            # print " 'Special' case after midnight", oh

        if is_open is not False:
            return oh
    return False