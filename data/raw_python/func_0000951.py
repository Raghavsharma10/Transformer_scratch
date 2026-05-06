def abstract_json(soup):
    """abstract in article json format"""
    abstract_tags = raw_parser.abstract(soup)
    abstract_json = None
    for tag in abstract_tags:
        if tag.get("abstract-type") is None:
            abstract_json = render_abstract_json(tag)
    return abstract_json