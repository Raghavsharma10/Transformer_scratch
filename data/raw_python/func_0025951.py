def nth_day_of_month(n, weekday, month, year):
    """
    Return (year, month, day) tuple that represents nth weekday of month in year.
    If n==0, returns last weekday of month. Weekdays: Monday=0
    """
    if not (0 <= n <= 5):
        raise IndexError("Nth day of month must be 0-5. Received: {}".format(n))

    if not (0 <= weekday <= 6):
        raise IndexError("Weekday must be 0-6")

    firstday, daysinmonth = calendar.monthrange(year, month)

    # Get first WEEKDAY of month
    first_weekday_of_kind = 1 + (weekday - firstday) % 7

    if n == 0:
        # find last weekday of kind, which is 5 if these conditions are met, else 4
        if first_weekday_of_kind in [1, 2, 3] and first_weekday_of_kind + 28 < daysinmonth:
            n = 5
        else:
            n = 4

    day = first_weekday_of_kind + ((n - 1) * 7)

    if day > daysinmonth:
        raise IndexError("No {}th day of month {}".format(n, month))

    return (year, month, day)