def digest_json(soup):
    """digest in article json format"""
    abstract_tags = raw_parser.abstract(soup, abstract_type="executive-summary")
    abstract_json = None
    for tag in abstract_tags:
        abstract_json = render_abstract_json(tag)
    return abstract_json