def pub_date(soup):
    """
    Return the publishing date in struct format
    pub_date_date, pub_date_day, pub_date_month, pub_date_year, pub_date_timestamp
    Default date_type is pub
    """
    pub_date = first(raw_parser.pub_date(soup, date_type="pub"))
    if pub_date is None:
        pub_date = first(raw_parser.pub_date(soup, date_type="publication"))
    if pub_date is None:
        return None
    (day, month, year) = ymd(pub_date)
    return date_struct(year, month, day)