def category(soup):
    """
    Find the category from subject areas
    """
    category = []

    tags = raw_parser.category(soup)
    for tag in tags:
        category.append(node_text(tag))

    return category