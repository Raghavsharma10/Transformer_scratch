def collection_year(soup):
    """
    Pub date of type collection will hold a year element for VOR articles
    """
    pub_date = first(raw_parser.pub_date(soup, pub_type="collection"))
    if not pub_date:
        pub_date = first(raw_parser.pub_date(soup, date_type="collection"))
    if not pub_date:
        return None

    year = None
    year_tag = raw_parser.year(pub_date)
    if year_tag:
        year = int(node_text(year_tag))

    return year