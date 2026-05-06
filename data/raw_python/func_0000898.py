def version_history(soup, html_flag=True):
    "extract the article version history details"
    convert = lambda xml_string: xml_to_html(html_flag, xml_string)
    version_history = []
    related_object_tags = raw_parser.related_object(raw_parser.article_meta(soup))
    for tag in related_object_tags:
        article_version = OrderedDict()
        date_tag = first(raw_parser.date(tag))
        if date_tag:
            copy_attribute(date_tag.attrs, 'date-type', article_version, 'version')
            (day, month, year) = ymd(date_tag)
            article_version['day'] = day
            article_version['month'] = month
            article_version['year'] = year
            article_version['date'] = date_struct_nn(year, month, day)
        copy_attribute(tag.attrs, 'xlink:href', article_version, 'xlink_href')
        set_if_value(article_version, "comment",
                     convert(node_contents_str(first(raw_parser.comment(tag)))))
        version_history.append(article_version)
    return version_history