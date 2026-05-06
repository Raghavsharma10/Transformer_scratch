def full_keywords(soup):
    "author keywords list including inline tags, such as italic"
    if not raw_parser.author_keywords(soup):
        return []
    return list(map(node_contents_str, raw_parser.author_keywords(soup)))