def get_months_apart(d1, d2):
    """
    Get amount of months between dates
    http://stackoverflow.com/a/4040338
    """

    return (d1.year - d2.year)*12 + d1.month - d2.month