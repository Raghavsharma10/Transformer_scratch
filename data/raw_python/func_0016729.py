def new_expiry(days=DEFAULT_PASTE_LIFETIME_DAYS):
    """Return an expiration `days` in the future"""
    now = delorean.Delorean()
    return now + datetime.timedelta(days=days)