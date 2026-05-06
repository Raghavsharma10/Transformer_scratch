def run_keybind_callbacks(e):
    """
    A function that intercepts all key press/release events, and runs
    their corresponding callback functions. Nothing much to see here, except
    that we must mask out the trivial modifiers from the state in order to
    find the right callback.
    Callbacks are called in the order that they have been added. (FIFO.)
    :param e: A Key{Press,Release} event.
    :type e: xcb.xproto.Key{Press,Release}Event
    :rtype: bool True if the callback was serviced
    """
    kc, mods = e.detail, e.state
    for mod in TRIVIAL_MODS:
        mods &= ~mod

    key = (e.event, mods, kc)
    serviced = False
    for cb in __keybinds.get(key, []):
        try:
            cb(e)
            serviced = True
        except TypeError:
            cb()
    return serviced