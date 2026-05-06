def set_metric(slug, value, category=None, expire=None, date=None):
    """Create/Increment a metric."""
    get_r().set_metric(slug, value, category=category, expire=expire, date=date)