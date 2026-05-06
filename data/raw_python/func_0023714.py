def ungrab_key(conn, wid, modifiers, key):
    """
    Ungrabs a key that was grabbed by ``grab_key``. Similarly, it will return
    True on success and False on failure.

    When ungrabbing a key, the parameters to this function should be
    *precisely* the same as the parameters to ``grab_key``.

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
            conn.core.UngrabKeyChecked(key, wid, modifiers | mod).check()

        return True
    except xproto.BadAccess:
        return False