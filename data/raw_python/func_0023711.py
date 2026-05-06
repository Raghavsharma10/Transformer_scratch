def get_keys_to_mods(conn):
    """
    Fetches and creates the keycode -> modifier mask mapping. Typically, you
    shouldn't have to use this---xpybutil will keep this up to date if it
    changes.

    This function may be useful in that it should closely replicate the output
    of the ``xmodmap`` command. For example:

     ::

        keymods = get_keys_to_mods()
        for kc in sorted(keymods, key=lambda kc: keymods[kc]):
            print keymods[kc], hex(kc), get_keysym_string(get_keysym(kc))

    Which will very closely replicate ``xmodmap``. I'm not getting precise
    results quite yet, but I do believe I'm getting at least most of what
    matters. (i.e., ``xmodmap`` returns valid keysym strings for some that
    I cannot.)

    :return: A dict mapping from keycode to modifier mask.
    :rtype: dict
    """
    mm = xproto.ModMask
    modmasks = [mm.Shift, mm.Lock, mm.Control,
                mm._1, mm._2, mm._3, mm._4, mm._5] # order matters

    mods = conn.core.GetModifierMapping().reply()

    res = {}
    keyspermod = mods.keycodes_per_modifier
    for mmi in range(0, len(modmasks)):
        row = mmi * keyspermod
        for kc in mods.keycodes[row:row + keyspermod]:
            res[kc] = modmasks[mmi]

    return res