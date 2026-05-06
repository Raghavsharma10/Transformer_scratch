def get_cache_key(page, language):
    """
    Create the cache key for the current page and language
    """
    from cms.cache import _get_cache_key
    try:
        site_id = page.node.site_id
    except AttributeError:  # CMS_3_4
        site_id = page.site_id
    return _get_cache_key('page_meta', page, language, site_id)