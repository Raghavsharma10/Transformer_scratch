def parse_tibia_date(date_str) -> Optional[datetime.date]:
    """Parses a date from the format used in Tibia.com

    Accepted format:

    - ``MMM DD YYYY``, e.g. ``Jul 23 2015``

    Parameters
    -----------
    date_str: :class:`str`
        The date as represented in Tibia.com

    Returns
    -----------
    :class:`datetime.date`, optional
        The represented date."""
    try:
        t = datetime.datetime.strptime(date_str.strip(), "%b %d %Y")
        return t.date()
    except (ValueError, AttributeError):
        return None