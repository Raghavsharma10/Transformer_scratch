def get_keyboard_mapping(conn):
    """
    Return a keyboard mapping cookie that can be used to fetch the table of
    keysyms in the current X environment.

    :rtype: xcb.xproto.GetKeyboardMappingCookie
    """
    mn, mx = get_min_max_keycode(conn)

    return conn.core.GetKeyboardMapping(mn, mx - mn + 1)