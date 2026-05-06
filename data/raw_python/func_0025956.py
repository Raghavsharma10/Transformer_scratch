def put(text, cbname):
    """ Put the given string into the given clipboard. """
    global _lastSel
    _checkTkInit()
    if cbname == 'CLIPBOARD':
        _theRoot.clipboard_clear()
        if text:
            # for clipboard_append, kwds can be -displayof, -format, or -type
            _theRoot.clipboard_append(text)
        return
    if cbname == 'PRIMARY':
        _lastSel = text
        _theRoot.selection_handle(ch_handler, selection='PRIMARY')
        # we need to claim/own it so that ch_handler is used
        _theRoot.selection_own(selection='PRIMARY')
        # could add command arg for a func to be called when we lose ownership
        return
    raise RuntimeError("Unexpected clipboard name: "+str(cbname))