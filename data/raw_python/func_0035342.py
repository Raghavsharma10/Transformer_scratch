def render_date(date, tz=pytz.utc, fmt=_FULL_OUTPUT_FORMAT):
    """Format the given date for output. The local time render of the given
    date is done using the given timezone."""
    local = date.astimezone(tz)
    ts = __date_to_millisecond_ts(date)
    return fmt.format(
            ts=ts,
            utc=date.strftime(_DATE_FORMAT),
            millis=ts % 1000,
            utc_tz=date.strftime(_TZ_FORMAT),
            local=local.strftime(_DATE_FORMAT),
            local_tz=local.strftime(_TZ_FORMAT),
            delta=render_delta_from_now(date))