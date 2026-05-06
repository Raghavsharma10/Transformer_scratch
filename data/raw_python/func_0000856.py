def remove_doi_paragraph(tags):
    "Given a list of tags, only return those whose text doesn't start with 'DOI:'"
    p_tags = list(filter(lambda tag: not starts_with_doi(tag), tags))
    p_tags = list(filter(lambda tag: not paragraph_is_only_doi(tag), p_tags))
    return p_tags