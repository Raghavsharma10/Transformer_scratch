def log_vacation_days():
    """ Sum and report taken days off. """
    days_off = get_days_off(rc.read())
    pretty_days = map(lambda day: day.strftime('%a %b %d %Y'), days_off)
    for day in pretty_days:
        print(day)