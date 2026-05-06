def author_contributions(soup, fntype_filter):
    """
    Find the fn tags included in the competing interest
    """

    author_contributions_section = extract_nodes(soup, "fn-group", attr="content-type", value="author-contribution")
    if not author_contributions_section:
        return None
    fn = extract_nodes(first(author_contributions_section), "fn")
    cons = footnotes(fn, fntype_filter)

    return cons