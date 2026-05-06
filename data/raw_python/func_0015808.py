def get_related_with_scores(page):
    """
    Returns list of related tuples (Entry instance, score) for
    specified page.

    :param page: the page instance.
    :rtype: list.
    """
    related = []
    entry   = Entry.get_for_model(page)

    if entry:
        related = entry.related_with_scores

    return related