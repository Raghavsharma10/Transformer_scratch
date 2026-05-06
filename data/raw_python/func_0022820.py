def _process_key(evt):
    """Helper to convert from wx keycode to vispy keycode"""
    key = evt.GetKeyCode()
    if key in KEYMAP:
        return KEYMAP[key], ''
    if 97 <= key <= 122:
        key -= 32
    if key >= 32 and key <= 127:
        return keys.Key(chr(key)), chr(key)
    else:
        return None, None