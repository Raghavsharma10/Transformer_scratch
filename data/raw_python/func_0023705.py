def parse_keystring(conn, key_string):
    """
    A utility function to turn strings like 'Mod1+Mod4+a' into a pair
    corresponding to its modifiers and keycode.

    :param key_string: String starting with zero or more modifiers followed
                       by exactly one key press.

                       Available modifiers: Control, Mod1, Mod2, Mod3, Mod4,
                       Mod5, Shift, Lock
    :type key_string: str
    :return: Tuple of modifier mask and keycode
    :rtype: (mask, int)
    """
    # FIXME this code is temporary hack, requires better abstraction
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtCore import Qt
    from .qt_keycodes import KeyTbl, ModsTbl

    keysequence = QKeySequence(key_string)
    ks = keysequence[0]

    # Calculate the modifiers
    mods = Qt.NoModifier
    qtmods = Qt.NoModifier
    modifiers = 0
    if (ks & Qt.ShiftModifier == Qt.ShiftModifier):
        mods |= ModsTbl.index(Qt.ShiftModifier)
        qtmods |= Qt.ShiftModifier.real
        modifiers |= getattr(xproto.KeyButMask, "Shift", 0)
    if (ks & Qt.AltModifier == Qt.AltModifier):
        mods |= ModsTbl.index(Qt.AltModifier)
        qtmods |= Qt.AltModifier.real
        modifiers |= getattr(xproto.KeyButMask, "Mod1", 0)
    if (ks & Qt.ControlModifier == Qt.ControlModifier):
        mods |= ModsTbl.index(Qt.ControlModifier)
        qtmods |= Qt.ControlModifier.real
        modifiers |= getattr(xproto.KeyButMask, "Control", 0)

    # Calculate the keys
    qtkeys = ks ^ qtmods
    key = QKeySequence(Qt.Key(qtkeys)).toString().lower()
    keycode = lookup_string(conn, key)
    return modifiers, keycode

    # Fallback logic
    modifiers = 0
    keycode = None

    key_string = "Shift+Control+A"
    for part in key_string.split('+'):
        if hasattr(xproto.KeyButMask, part):
            modifiers |= getattr(xproto.KeyButMask, part)
        else:
            if len(part) == 1:
                part = part.lower()
            keycode = lookup_string(conn, part)

    return modifiers, keycode