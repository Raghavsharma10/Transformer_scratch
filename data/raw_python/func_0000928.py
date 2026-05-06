def full_author_notes(soup, fntype_filter=None):
    """
    Find the fn tags included in author-notes
    """
    notes = []

    author_notes_section = raw_parser.author_notes(soup)
    if author_notes_section:
        fn_nodes = raw_parser.fn(author_notes_section)
        notes = footnotes(fn_nodes, fntype_filter)

    return notes