def get_keyboard_mapping_unchecked(conn):
    """
    Return an unchecked keyboard mapping cookie that can be used to fetch the
    table of keysyms in the current X environment.

    :rtype: xcb.xproto.GetKeyboardMappingCookie
    """
    mn, mx = get_min_max_keycode()

    return conn.core.GetKeyboardMappingUnchecked(mn, mx - mn + 1)