def author_line(soup):
    """take preferred names from authors json and format them into an author line"""
    author_line = None
    authors_json_data = authors_json(soup)
    author_names = extract_author_line_names(authors_json_data)
    if len(author_names) > 0:
        author_line = format_author_line(author_names)
    return author_line