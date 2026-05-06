def _get_mods(evt):
    """Helper to extract list of mods from event"""
    mods = []
    mods += [keys.CONTROL] if evt.ControlDown() else []
    mods += [keys.ALT] if evt.AltDown() else []
    mods += [keys.SHIFT] if evt.ShiftDown() else []
    mods += [keys.META] if evt.MetaDown() else []
    return mods