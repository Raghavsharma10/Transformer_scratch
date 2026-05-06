def __regrab(changes):
    """
    Takes a dictionary of changes (mapping old keycode to new keycode) and
    regrabs any keys that have been changed with the updated keycode.

    :param changes: Mapping of changes from old keycode to new keycode.
    :type changes: dict
    :rtype: void
    """
    for wid, mods, kc in __keybinds.keys():
        if kc in changes:
            ungrab_key(wid, mods, kc)
            grab_key(wid, mods, changes[kc])

            old = (wid, mods, kc)
            new = (wid, mods, changes[kc])
            __keybinds[new] = __keybinds[old]
            del __keybinds[old]