def get_keysym(conn, keycode, col=0, kbmap=None):
    """
    Get the keysym associated with a particular keycode in the current X
    environment. Although we get a list of keysyms from X in
    'get_keyboard_mapping', this list is really a table with
    'keysys_per_keycode' columns and ``mx - mn`` rows (where ``mx`` is the
    maximum keycode and ``mn`` is the minimum keycode).

    Thus, the index for a keysym given a keycode is:
    ``(keycode - mn) * keysyms_per_keycode + col``.

    In most cases, setting ``col`` to 0 will work.

    Witness the utter complexity:
    http://tronche.com/gui/x/xlib/input/keyboard-encoding.html

    You may also pass in your own keyboard mapping using the ``kbmap``
    parameter, but xpybutil maintains an up-to-date version of this so you
    shouldn't have to.

    :param keycode: A physical key represented by an integer.
    :type keycode: int
    :param col: The column in the keysym table to use.
                Unless you know what you're doing, just use 0.
    :type col: int
    :param kbmap: The keyboard mapping to use.
    :type kbmap: xcb.xproto.GetKeyboardMapingReply
    """
    if kbmap is None:
        kbmap = __kbmap

    mn, mx = get_min_max_keycode(conn)
    per = kbmap.keysyms_per_keycode
    ind = (keycode - mn) * per + col

    return kbmap.keysyms[ind]