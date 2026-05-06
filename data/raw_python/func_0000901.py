def subject_area(soup):
    """
    Find the subject areas from article-categories subject tags
    """
    subject_area = []

    tags = raw_parser.subject_area(soup)
    for tag in tags:
        subject_area.append(node_text(tag))

    return subject_area