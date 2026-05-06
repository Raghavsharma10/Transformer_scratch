def alert(text='', title='', button='OK'):
    """Displays a simple message box with text and a single OK button. Returns the text of the button clicked on."""
    messageBoxFunc(0, text, title, MB_OK | MB_SETFOREGROUND | MB_TOPMOST)
    return button