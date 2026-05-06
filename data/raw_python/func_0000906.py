def pub_dates(soup):
    """
    return a list of all the pub dates
    """
    pub_dates = []
    tags = raw_parser.pub_date(soup)
    for tag in tags:
        pub_date = OrderedDict()
        copy_attribute(tag.attrs, 'publication-format', pub_date)
        copy_attribute(tag.attrs, 'date-type', pub_date)
        copy_attribute(tag.attrs, 'pub-type', pub_date)
        for tag_attr in ["date-type", "pub-type"]:
            if tag_attr in tag.attrs:
                (day, month, year) = ymd(tag)
                pub_date['day'] = day
                pub_date['month'] = month
                pub_date['year'] = year
                pub_date['date'] = date_struct_nn(year, month, day)
        pub_dates.append(pub_date)
    return pub_dates