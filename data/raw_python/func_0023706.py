def lookup_string(conn, kstr):
    """
    Finds the keycode associated with a string representation of a keysym.

    :param kstr: English representation of a keysym.
    :return: Keycode, if one exists.
    :rtype: int
    """
    if kstr in keysyms:
        return get_keycode(conn, keysyms[kstr])
    elif len(kstr) > 1 and kstr.capitalize() in keysyms:
        return get_keycode(conn, keysyms[kstr.capitalize()])

    return None