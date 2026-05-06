def copy(string):
    """Copy given string into system clipboard.
    """
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(string)
    win32clipboard.CloseClipboard()