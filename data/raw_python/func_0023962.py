def get_cached_menus():
    """Return the menus from the cache or generate them if needed."""
    items = cache.get(CACHE_KEY)
    if items is None:
        menu = generate_menu()
        cache.set(CACHE_KEY, menu.items)
    else:
        menu = Menu(items)
    return menu