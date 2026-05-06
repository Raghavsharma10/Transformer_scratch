def load_settings_sizes():
    """
    Load sizes from settings or fallback to the module constants
    """
    page_size = AGNOCOMPLETE_DEFAULT_PAGESIZE
    settings_page_size = getattr(
        settings, 'AGNOCOMPLETE_DEFAULT_PAGESIZE', None)
    page_size = settings_page_size or page_size

    page_size_min = AGNOCOMPLETE_MIN_PAGESIZE
    settings_page_size_min = getattr(
        settings, 'AGNOCOMPLETE_MIN_PAGESIZE', None)
    page_size_min = settings_page_size_min or page_size_min

    page_size_max = AGNOCOMPLETE_MAX_PAGESIZE
    settings_page_size_max = getattr(
        settings, 'AGNOCOMPLETE_MAX_PAGESIZE', None)
    page_size_max = settings_page_size_max or page_size_max

    # Query sizes
    query_size = AGNOCOMPLETE_DEFAULT_QUERYSIZE
    settings_query_size = getattr(
        settings, 'AGNOCOMPLETE_DEFAULT_QUERYSIZE', None)
    query_size = settings_query_size or query_size

    query_size_min = AGNOCOMPLETE_MIN_QUERYSIZE
    settings_query_size_min = getattr(
        settings, 'AGNOCOMPLETE_MIN_QUERYSIZE', None)
    query_size_min = settings_query_size_min or query_size_min

    return (
        page_size, page_size_min, page_size_max,
        query_size, query_size_min,
    )