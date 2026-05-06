def date_range_filter(range_name):
    """Create a filter from a named date range."""

    filter_days = list(filter(
        lambda time: time["label"] == range_name,
        settings.CUSTOM_SEARCH_TIME_PERIODS))
    num_days = filter_days[0]["days"] if len(filter_days) else None

    if num_days:
        dt = timedelta(num_days)
        start_time = timezone.now() - dt
        return Range(published={"gte": start_time})
    return MatchAll()