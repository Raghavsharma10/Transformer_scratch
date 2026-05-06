def metric(slug, num=1, category=None, expire=None, date=None):
    """Create/Increment a metric."""
    get_r().metric(slug, num=num, category=category, expire=expire, date=date)