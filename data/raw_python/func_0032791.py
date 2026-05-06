def weekly(date=datetime.date.today()):
    """
    Weeks start are fixes at Monday for now.
    """
    return date - datetime.timedelta(days=date.weekday())