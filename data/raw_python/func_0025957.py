def get(cbname):
    """ Get the contents of the given clipboard. """
    _checkTkInit()
    if cbname == 'PRIMARY':
        try:
            return _theRoot.selection_get(selection='PRIMARY')
        except:
            return None
    if cbname == 'CLIPBOARD':
        try:
            return _theRoot.selection_get(selection='CLIPBOARD')
        except:
            return None
    raise RuntimeError("Unexpected clipboard name: "+str(cbname))