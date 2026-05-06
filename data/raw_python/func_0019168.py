def iso_day_to_weekday(d):
    """
    Returns the weekday's name given a ISO weekday number;
    "today" if today is the same weekday.
    """
    if int(d) == utils.get_now().isoweekday():
        return _("today")
    for w in WEEKDAYS:
        if w[0] == int(d):
            return w[1]