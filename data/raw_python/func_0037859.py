def popular_content(**kwargs):
    """
    Use the get_popular_ids() to retrieve trending content objects.
    Return recent content on failure.
    """
    limit = kwargs.get("limit", DEFAULT_LIMIT)
    popular_ids = get_popular_ids(limit=limit)
    if not popular_ids:
        # Return most recent content
        return Content.search_objects.search().extra(size=limit)
    return Content.search_objects.search().filter(es_filter.Ids(values=popular_ids))