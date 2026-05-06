def bind_global_key(conn, event_type, key_string, cb):
    """
    An alias for ``bind_key(event_type, ROOT_WINDOW, key_string, cb)``.
    :param event_type: Either 'KeyPress' or 'KeyRelease'.
    :type event_type: str
    :param key_string: A string of the form 'Mod1-Control-a'.
                       Namely, a list of zero or more modifiers separated by
                       '-', followed by a single non-modifier key.
    :type key_string: str
    :param cb: A first class function with no parameters.
    :type cb: function
    :return: True if the binding was successful, False otherwise.
    :rtype: bool
    """
    root = conn.get_setup().roots[0].root
    return bind_key(conn, event_type, root, key_string, cb)