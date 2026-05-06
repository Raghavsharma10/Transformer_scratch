def parse_datetime(record: str) -> Optional[datetime]:
    """ Parse a datetime string into a python datetime object """
    # NEM defines Date8, DateTime12 and DateTime14
    format_strings = {8: '%Y%m%d', 12: '%Y%m%d%H%M', 14: '%Y%m%d%H%M%S'}
    if record == '':
        return None
    return datetime.strptime(record.strip(),
                                          format_strings[len(record.strip())])