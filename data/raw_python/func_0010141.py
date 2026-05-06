def cache_page(page_cache, page_hash, cache_size):
    """Add a page to the page cache."""
    page_cache.append(page_hash)
    if len(page_cache) > cache_size:
        page_cache.pop(0)