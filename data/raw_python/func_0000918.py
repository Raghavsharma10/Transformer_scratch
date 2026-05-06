def contrib_phone(contrib_tag):
    """
    Given a contrib tag, look for an phone tag
    """
    phone = None
    if raw_parser.phone(contrib_tag):
        phone = first(raw_parser.phone(contrib_tag)).text
    return phone