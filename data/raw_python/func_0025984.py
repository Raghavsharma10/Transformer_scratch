def init_tk_default_root(withdraw=True):

    """ In case the _default_root value is required, you may
    safely call this ahead of time to ensure that it has been
    initialized.  If it has already been, this is a no-op.
    """
    if not capable.OF_GRAPHICS:
        raise RuntimeError("Cannot run this command without graphics")

    if not TKNTR._default_root: # TKNTR imported above
        junk = TKNTR.Tk()

    # tkinter._default_root is now populated (== junk)
    retval = TKNTR._default_root
    if withdraw and retval:
        retval.withdraw()

    return retval