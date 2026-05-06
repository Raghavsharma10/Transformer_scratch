def collab_to_group_author_key_map(authors):
    """compile a map of author collab to group-author-key"""
    collab_map = {}
    for author in authors:
        if author.get("collab"):
            collab_map[author.get("collab")] = author.get("group-author-key")
    return collab_map