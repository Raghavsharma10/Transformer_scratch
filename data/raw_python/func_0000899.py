def article_id_list(soup):
    """return a list of article-id data"""
    id_list = []
    for article_id_tag in raw_parser.article_id(soup):
        id_details = OrderedDict()
        set_if_value(id_details, "type", article_id_tag.get("pub-id-type"))
        set_if_value(id_details, "value", article_id_tag.text)
        set_if_value(id_details, "assigning-authority", article_id_tag.get("assigning-authority"))
        id_list.append(id_details)
    return id_list