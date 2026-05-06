def format_author_line(author_names):
    """authorLine format depends on if there is 1, 2 or more than 2 authors"""
    author_line = None
    if not author_names:
        return author_line
    if len(author_names) <= 2:
        author_line = ", ".join(author_names)
    elif len(author_names) > 2:
        author_line = author_names[0] + " et al."
    return author_line