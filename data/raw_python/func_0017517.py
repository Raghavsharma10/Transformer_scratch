def confirm(text='', title='', buttons=['OK', 'Cancel']):
    """Displays a message box with OK and Cancel buttons. Number and text of buttons can be customized. Returns the text of the button clicked on."""
    retVal = messageBoxFunc(0, text, title, MB_OKCANCEL | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST)
    if retVal == 1 or len(buttons) == 1:
        return buttons[0]
    elif retVal == 2:
        return buttons[1]
    else:
        assert False, 'Unexpected return value from MessageBox: %s' % (retVal)