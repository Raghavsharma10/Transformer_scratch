def openParametersDialog(params, title=None):
    '''
    Opens a dialog to enter parameters.
    Parameters are passed as a list of Parameter objects
    Returns a dict with param names as keys and param values as values
    Returns None if the dialog was cancelled
    '''
    QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
    dlg = ParametersDialog(params, title)
    dlg.exec_()
    QApplication.restoreOverrideCursor()
    return dlg.values