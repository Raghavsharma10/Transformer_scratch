def ymd(soup):
    """
    Get the year, month and day from child tags
    """
    day = node_text(raw_parser.day(soup))
    month = node_text(raw_parser.month(soup))
    year = node_text(raw_parser.year(soup))
    return (day, month, year)