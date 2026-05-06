def date_range(start_date, end_date, increment, period):
    """
    Generate `date` objects between `start_date` and `end_date` in `increment`
    `period` intervals.
    """
    next = start_date
    delta = relativedelta.relativedelta(**{period:increment})
    while next <= end_date:
        yield next
        next += delta