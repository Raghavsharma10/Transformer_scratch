def page_menus(root_menu):
    """Add page menus."""
    for page in Page.objects.filter(include_in_menu=True):
        path = page.get_path()
        menu = path[0] if len(path) > 1 else None
        try:
            root_menu.add_item(page.name, page.get_absolute_url(), menu=menu)
        except MenuError as e:
            logger.error("Bad menu item %r for page with slug %r."
                         % (e, page.slug))