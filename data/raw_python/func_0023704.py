def bind_key(conn, event_type, wid, key_string, cb):
    """
    Binds a function ``cb`` to a particular key press ``key_string`` on a
    window ``wid``. Whether it's a key release or key press binding is
    determined by ``event_type``.
    ``bind_key`` will automatically hook into the ``event`` module's dispatcher,
    so that if you're using ``event.main()`` for your main loop, everything
    will be taken care of for you.
    :param event_type: Either 'KeyPress' or 'KeyRelease'.
    :type event_type: str
    :param wid: The window to bind the key grab to.
    :type wid: int
    :param key_string: A string of the form 'Mod1-Control-a'.
                       Namely, a list of zero or more modifiers separated by
                       '-', followed by a single non-modifier key.
    :type key_string: str
    :param cb: A first class function with no parameters.
    :type cb: function
    :return: True if the binding was successful, False otherwise.
    :rtype: bool
    """
    assert event_type in ('KeyPress', 'KeyRelease')

    mods, kc = parse_keystring(conn, key_string)
    key = (wid, mods, kc)

    if not kc:
        print("Could not find a keycode for " + key_string)
        return False

    if not __keygrabs[key] and not grab_key(conn, wid, mods, kc):
        return False

    __keybinds[key].append(cb)
    __keygrabs[key] += 1

    return True