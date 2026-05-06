def competing_interests(soup, fntype_filter):
    """
    Find the fn tags included in the competing interest
    """

    competing_interests_section = extract_nodes(soup, "fn-group", attr="content-type", value="competing-interest")
    if not competing_interests_section:
        return None
    fn = extract_nodes(first(competing_interests_section), "fn")
    interests = footnotes(fn, fntype_filter)

    return interests