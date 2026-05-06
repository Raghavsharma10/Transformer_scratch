def paginator(limit, offset, record_count, base_uri, page_nav_tpl='&limit={}&offset={}'):
    """Compute pagination info for collection filtering.

    Args:
        limit (int): Collection filter limit.
        offset (int): Collection filter offset.
        record_count (int): Collection filter total record count.
        base_uri (str): Collection filter base uri (without limit, offset)
        page_nav_tpl (str): Pagination template.

    Returns:
        A mapping of pagination info.
    """

    total_pages = int(math.ceil(record_count / limit))

    next_cond = limit + offset <= record_count
    prev_cond = offset >= limit

    next_page = base_uri + page_nav_tpl.format(limit, offset + limit) if next_cond else None

    prev_page = base_uri + page_nav_tpl.format(limit, offset - limit) if prev_cond else None

    return OrderedDict([
        ('total_count', record_count),
        ('total_pages', total_pages),
        ('next_page', next_page),
        ('prev_page', prev_page)
    ])