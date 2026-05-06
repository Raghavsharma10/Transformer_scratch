def pp_event(seq):
    """Returns pretty representation of an Event or keypress"""

    if isinstance(seq, Event):
        return str(seq)

    # Get the original sequence back if seq is a pretty name already
    rev_curses = dict((v, k) for k, v in CURSES_NAMES.items())
    rev_curtsies = dict((v, k) for k, v in CURTSIES_NAMES.items())
    if seq in rev_curses:
        seq = rev_curses[seq]
    elif seq in rev_curtsies:
        seq = rev_curtsies[seq]

    pretty = curtsies_name(seq)
    if pretty != seq:
        return pretty
    return repr(seq).lstrip('u')[1:-1]