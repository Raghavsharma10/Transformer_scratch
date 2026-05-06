def update_keyboard_mapping(conn, e):
    """
    Whenever the keyboard mapping is changed, this function needs to be called
    to update xpybutil's internal representing of the current keysym table.
    Indeed, xpybutil will do this for you automatically.

    Moreover, if something is changed that affects the current keygrabs,
    xpybutil will initiate a regrab with the changed keycode.

    :param e: The MappingNotify event.
    :type e: xcb.xproto.MappingNotifyEvent
    :rtype: void
    """
    global __kbmap, __keysmods

    newmap = get_keyboard_mapping(conn).reply()

    if e is None:
        __kbmap = newmap
        __keysmods = get_keys_to_mods(conn)
        return

    if e.request == xproto.Mapping.Keyboard:
        changes = {}
        for kc in range(*get_min_max_keycode(conn)):
            knew = get_keysym(kc, kbmap=newmap)
            oldkc = get_keycode(conn, knew)
            if oldkc != kc:
                changes[oldkc] = kc

        __kbmap = newmap
        __regrab(changes)
    elif e.request == xproto.Mapping.Modifier:
        __keysmods = get_keys_to_mods()