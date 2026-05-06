def format_datetime(time):
    """
    Formats a date, converting the time to the user timezone if one is specified
    """
    user_time_zone = timezone.get_current_timezone()
    if time.tzinfo is None:
        time = time.replace(tzinfo=pytz.utc)
        user_time_zone = pytz.timezone(getattr(settings, 'USER_TIME_ZONE', 'GMT'))

    time = time.astimezone(user_time_zone)
    return time.strftime("%b %d, %Y %H:%M")