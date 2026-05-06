def grab_key(conn, wid, modifiers, key):
    """
    Grabs a key for a particular window and a modifiers/key value.
    If the grab was successful, return True. Otherwise, return False.
    If your client is grabbing keys, it is useful to notify the user if a
    key wasn't grabbed. Keyboard shortcuts not responding is disorienting!

    Also, this function will grab several keys based on varying modifiers.
    Namely, this accounts for all of the "trivial" modifiers that may have
    an effect on X events, but probably shouldn't effect key grabbing. (i.e.,
    whether num lock or caps lock is on.)

    N.B. You should probably be using 'bind_key' or 'bind_global_key' instead.

    :param wid: A window identifier.
    :type wid: int
    :param modifiers: A modifier mask.
    :type modifiers: int
    :param key: A keycode.
    :type key: int
    :rtype: bool
    """
    try:
        for mod in TRIVIAL_MODS:
            conn.core.GrabKeyChecked(True, wid, modifiers | mod, key, GM.Async,
                                     GM.Async).check()

        return True
    except xproto.BadAccess:
        return False