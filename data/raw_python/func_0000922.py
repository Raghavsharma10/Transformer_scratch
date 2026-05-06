def authors_non_byline(soup, detail="full"):
    """Non-byline authors for group author members"""
    # Get a filtered list of contributors, in order to get their group-author-id
    contrib_type = "author non-byline"
    contributors_ = contributors(soup, detail)
    non_byline_authors = [author for author in contributors_ if author.get('type', None) == contrib_type]

    # Then renumber their position attribute
    position = 1
    for author in non_byline_authors:
        author["position"] = position
        position = position + 1
    return non_byline_authors