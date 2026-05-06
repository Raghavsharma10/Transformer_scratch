def _validate_date_str(str_):
    """Validate str as a date and return string version of date"""

    if not str_:
        return None

    # Convert to datetime so we can validate it's a real date that exists then
    # convert it back to the string.
    try:
        date = datetime.strptime(str_, DATE_FMT)
    except ValueError:
        msg = 'Invalid date format, should be YYYY-MM-DD'
        raise argparse.ArgumentTypeError(msg)

    return date.strftime(DATE_FMT)