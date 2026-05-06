def author_notes(soup):
    """
    Find the fn tags included in author-notes
    """
    author_notes = []

    author_notes_section = raw_parser.author_notes(soup)
    if author_notes_section:
        fn_nodes = raw_parser.fn(author_notes_section)
        for tag in fn_nodes:
            if 'fn-type' in tag.attrs:
                if(tag['fn-type'] != 'present-address'):
                    author_notes.append(node_text(tag))

    return author_notes