def generate_menu():
    """Generate a new list of menus."""
    root_menu = Menu(list(copy.deepcopy(settings.WAFER_MENUS)))
    for dynamic_menu_func in settings.WAFER_DYNAMIC_MENUS:
        dynamic_menu_func = maybe_obj(dynamic_menu_func)
        dynamic_menu_func(root_menu)
    return root_menu