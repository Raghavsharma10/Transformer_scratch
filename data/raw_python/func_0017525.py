def confirm(text='', title='', buttons=[OK_TEXT, CANCEL_TEXT], root=None, timeout=None):
    """Displays a message box with OK and Cancel buttons. Number and text of buttons can be customized. Returns the text of the button clicked on."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    return _buttonbox(msg=text, title=title, choices=[str(b) for b in buttons], root=root, timeout=timeout)