def get_dates_in_period(start=None, top=None, step=1, step_dict={}):
    """Return a list of dates from the `start` to `top`."""

    delta = relativedelta(**step_dict) if step_dict else timedelta(days=step)

    start = start or datetime.today()
    top = top or start + delta
    dates = []
    current = start
    while current <= top:
        dates.append(current)
        current += delta
    return dates