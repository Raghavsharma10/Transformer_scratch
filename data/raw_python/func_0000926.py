def correspondence(soup):
    """
    Find the corresp tags included in author-notes
    for primary correspondence
    """
    correspondence = []

    author_notes_nodes = raw_parser.author_notes(soup)

    if author_notes_nodes:
        corresp_nodes = raw_parser.corresp(author_notes_nodes)
        for tag in corresp_nodes:
            correspondence.append(tag.text)

    return correspondence