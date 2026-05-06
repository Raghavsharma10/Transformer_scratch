def fill_date_range(start_date, end_date, date_format=None):
    """
    Function accepts start date, end date, and format (if dates are strings)
    and returns a list of Python dates.
    """

    if date_format:
        start_date = datetime.strptime(start_date, date_format).date()
        end_date = datetime.strptime(end_date, date_format).date()
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date)
        start_date = start_date + timedelta(days=1)
    return date_list