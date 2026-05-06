def show():
    """Try showing the most desirable GUI

    This function cycles through the currently registered
    graphical user interfaces, if any, and presents it to
    the user.

    """

    parent = None
    current = QtWidgets.QApplication.activeWindow()
    while current:
        parent = current
        current = parent.parent()

    window = (_discover_gui() or _show_no_gui)(parent)

    return window