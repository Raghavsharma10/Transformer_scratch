def get_modifiers(state):
    """
    Takes a ``state`` (typically found in key press or button press events)
    and returns a string list representation of the modifiers that were pressed
    when generating the event.

    :param state: Typically from ``some_event.state``.
    :return: List of modifier string representations.
    :rtype: [str]
    """
    ret = []

    if state & xproto.ModMask.Shift:
        ret.append('Shift')
    if state & xproto.ModMask.Lock:
        ret.append('Lock')
    if state & xproto.ModMask.Control:
        ret.append('Control')
    if state & xproto.ModMask._1:
        ret.append('Mod1')
    if state & xproto.ModMask._2:
        ret.append('Mod2')
    if state & xproto.ModMask._3:
        ret.append('Mod3')
    if state & xproto.ModMask._4:
        ret.append('Mod4')
    if state & xproto.ModMask._5:
        ret.append('Mod5')
    if state & xproto.KeyButMask.Button1:
        ret.append('Button1')
    if state & xproto.KeyButMask.Button2:
        ret.append('Button2')
    if state & xproto.KeyButMask.Button3:
        ret.append('Button3')
    if state & xproto.KeyButMask.Button4:
        ret.append('Button4')
    if state & xproto.KeyButMask.Button5:
        ret.append('Button5')

    return ret