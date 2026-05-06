def next_time_open(location):
    """
    Returns the next possible opening hours object, or (False, None)
    if location is currently open or there is no such object
    I.e. when is the company open for the next time?
    """
    if not is_open(location):
        now = get_now()
        now_time = datetime.time(now.hour, now.minute, now.second)
        found_opening_hours = False
        for i in range(8):
            l_weekday = (now.isoweekday() + i) % 7
            ohs = OpeningHours.objects.filter(company=location,
                                              weekday=l_weekday
                                              ).order_by('weekday',
                                                         'from_hour')

            if ohs.count():
                for oh in ohs:
                    future_now = now + datetime.timedelta(days=i)
                    # same day issue
                    tmp_now = datetime.datetime(future_now.year,
                                                future_now.month,
                                                future_now.day,
                                                oh.from_hour.hour,
                                                oh.from_hour.minute,
                                                oh.from_hour.second)
                    if tmp_now < now:
                        tmp_now = now  # be sure to set the bound correctly...
                    if is_open(location, now=tmp_now):
                        found_opening_hours = oh
                        break
                if found_opening_hours is not False:
                    return found_opening_hours, tmp_now
    return False, None