def localize_date(date, city):
    """ Localize date into city

    Date: datetime
    City: timezone city definitio. Example: 'Asia/Qatar', 'America/New York'..
    """
    local = pytz.timezone(city)
    local_dt = local.localize(date, is_dst=None)
    return local_dt