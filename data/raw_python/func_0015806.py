def get_related(page):
    """
    Returns list of related Entry instances for specified page.

    :param page: the page instance.
    :rtype: list.
    """
    related = []
    entry   = Entry.get_for_model(page)

    if entry:
        related = entry.related

    return related