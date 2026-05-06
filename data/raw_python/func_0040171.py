def _get_limit_and_offset(page, page_size):
    """Returns a 0-indexed offset and limit based on page and page_size for a MySQL query.
    """
    if page < 1:
        raise ValueError('page must be >= 1')
    limit = page_size
    offset = (page - 1) * page_size
    return limit, offset