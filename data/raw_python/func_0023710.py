def get_keycode(conn, keysym):
    """
    Given a keysym, find the keycode mapped to it in the current X environment.
    It is necessary to search the keysym table in order to do this, including
    all columns.

    :param keysym: An X keysym.
    :return: A keycode or None if one could not be found.
    :rtype: int
    """
    mn, mx = get_min_max_keycode(conn)
    cols = __kbmap.keysyms_per_keycode
    for i in range(mn, mx + 1):
        for j in range(0, cols):
            ks = get_keysym(conn, i, col=j)
            if ks == keysym:
                return i

    return None