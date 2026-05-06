def enum_menu(strs, menu=None, *args, **kwargs):
    """Enumerates the given list of strings into returned menu.

    **Params**:
      - menu (Menu) - Existing menu to append. If not provided, a new menu will
        be created.
    """
    if not menu:
        menu = Menu(*args, **kwargs)
    for s in strs:
        menu.enum(s)
    return menu